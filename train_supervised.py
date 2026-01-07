import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import os
import json
import pickle
import random
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from network import input_shape, load_model
from process_database import generate_records, play
from hparams import *

dataset_dir = Path(SL_DATASET_DIR)

def save_checkpoint(epoch, model_state, optimizer_state, scheduler_state):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
        'scheduler_state_dict': scheduler_state,
    }, SL_MODEL_PATH + ".checkpoint")
    
    with open(SL_CHECKPOINT_FILE, 'w') as f:
        json.dump({'epoch': epoch}, f)

def load_checkpoint():
    if os.path.exists(SL_CHECKPOINT_FILE):
        with open(SL_CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

def ensure_dataset_exists():
    if dataset_dir.exists() and any(dataset_dir.glob('chunk_*.pkl')):
        print(f"Dataset found in {dataset_dir}")
        return

    print(f"Creating dataset in {dataset_dir}...")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    records = generate_records()
    print(f"Found {len(records)} records.")

    history = []
    for i in tqdm(range(len(records))):
        history.extend(play(records[i]))

    print(f"Generated {len(history)} samples.")

    random.shuffle(history)
    
    split_idx = int(len(history) * (1 - SL_TEST_RATIO))
    train_history = history[:split_idx]
    test_history = history[split_idx:]
    
    print(f"Train/Test split: {len(train_history)} / {len(test_history)}")
    
    # Save Train Chunks
    for i in range(0, len(train_history), SL_CHUNK_SIZE):
        chunk = train_history[i:i + SL_CHUNK_SIZE]
        print(f"Processing train chunk {i // SL_CHUNK_SIZE + 1} / {(len(train_history) + SL_CHUNK_SIZE - 1) // SL_CHUNK_SIZE}...")
            
        output_path = dataset_dir / f"chunk_{i // SL_CHUNK_SIZE:03d}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(chunk, f)
            
    # Save Test Chunks
    for i in range(0, len(test_history), SL_CHUNK_SIZE):
        chunk = test_history[i:i + SL_CHUNK_SIZE]
        print(f"Processing test chunk {i // SL_CHUNK_SIZE + 1} / {(len(test_history) + SL_CHUNK_SIZE - 1) // SL_CHUNK_SIZE}...")
            
        output_path = dataset_dir / f"test_chunk_{i // SL_CHUNK_SIZE:03d}.pkl"
        with open(output_path, 'wb') as f:
            pickle.dump(chunk, f)
            
    print("Dataset creation complete.")

def load_chunk(file_path):
    with open(file_path, 'rb') as f:
        history = pickle.load(f)
    
    xs, y_policies, y_values = zip(*history)
    c, a, b = input_shape
    xs = np.array(xs).reshape(len(xs), c, a, b)
    y_policies = np.array(y_policies)
    y_values = np.array(y_values).reshape(-1, 1)

    xs_tensor = torch.FloatTensor(xs)
    y_policies_tensor = torch.FloatTensor(y_policies)
    y_values_tensor = torch.FloatTensor(y_values)

    return TensorDataset(xs_tensor, y_policies_tensor, y_values_tensor)

def evaluate_on_test(model, test_files, device, value_criterion):
    model.eval()
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_samples = 0
    
    with torch.no_grad():
        for file_path in test_files:
            print(f"    Evaluating on {file_path.name}...")
            dataset = load_chunk(file_path)
            dataloader = DataLoader(dataset, batch_size=SL_BATCH_SIZE, shuffle=False)
            
            for batch_x, batch_y_policy, batch_y_value in tqdm(dataloader, leave=False):
                batch_x = batch_x.to(device)
                batch_y_policy = batch_y_policy.to(device)
                batch_y_value = batch_y_value.to(device)
                
                pred_policy, pred_value = model(batch_x)
                
                # Summing loss manually to aggregate correctly
                policy_loss = -torch.sum(batch_y_policy * torch.log(pred_policy + 1e-8))
                value_loss = torch.sum((pred_value - batch_y_value) ** 2)
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_samples += batch_x.size(0)
    
    if total_samples == 0:
        return 0.0, 0.0
        
    return total_policy_loss / total_samples, total_value_loss / total_samples

def train_from_records():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name="train_supervised")

    # Prepare Data
    ensure_dataset_exists()
    train_files = list(dataset_dir.glob('chunk_*.pkl'))
    test_files = list(dataset_dir.glob('test_chunk_*.pkl'))
    print(f"Found {len(train_files)} train files and {len(test_files)} test files.")

    # Load Model
    model_path = './model/best.pth'
    if not os.path.exists(model_path):
        print("No existing model found, creating new one...")
        from network import dual_network
        dual_network()
    
    model = load_model(model_path)
    model = model.to(device)
    model.train()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=SL_LEARNING_RATE, weight_decay=1e-4)
    value_criterion = nn.MSELoss()
    
    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=SL_SCHEDULER_T0, T_mult=SL_SCHEDULER_TMULT)

    # Resume
    start_epoch = 0
    checkpoint_info = load_checkpoint()
    if checkpoint_info:
        start_epoch = checkpoint_info['epoch'] + 1
        if os.path.exists(SL_MODEL_PATH + ".checkpoint"):
            checkpoint = torch.load(SL_MODEL_PATH + ".checkpoint")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"Resuming from epoch {start_epoch}")

    # Train Loop
    iter = 0
    for epoch in range(start_epoch, SL_EPOCHS):
        print(f"Epoch {epoch + 1}/{SL_EPOCHS}")
        random.shuffle(train_files)
        
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        total_batches = 0
        
        for file_idx, file_path in enumerate(train_files):
            model.train() # Ensure train mode
            print(f"  Loading {file_path.name} ({file_idx + 1}/{len(train_files)})...")
            dataset = load_chunk(file_path)
            dataloader = DataLoader(dataset, batch_size=SL_BATCH_SIZE, shuffle=True)

            print(f"  Training on {len(dataset)} samples...")
            
            for batch_x, batch_y_policy, batch_y_value in tqdm(dataloader, leave=False):
                batch_x = batch_x.to(device)
                batch_y_policy = batch_y_policy.to(device)
                batch_y_value = batch_y_value.to(device)
                
                optimizer.zero_grad()
                pred_policy, pred_value = model(batch_x)
                
                policy_loss = -torch.sum(batch_y_policy * torch.log(pred_policy + 1e-8)) / batch_x.size(0)
                value_loss = value_criterion(pred_value, batch_y_value)
                
                total_loss = policy_loss + value_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_policy_loss += policy_loss.item()
                epoch_value_loss += value_loss.item()
                total_batches += 1
            
            # Evaluate after each file (User request: "test when one file training ends")
            test_policy_loss, test_value_loss = 0.0, 0.0
            if (iter + 1) % SL_TEST_INTERVAL == 0:
                print(f"  Running test evaluation...")
                test_policy_loss, test_value_loss = evaluate_on_test(model, test_files, device, value_criterion)
                print(f"  Test Results - Policy Loss: {test_policy_loss:.4f} | Value Loss: {test_value_loss:.4f}")

            if USE_WANDB:
                wandb_log = {
                    "epoch": epoch,
                    "policy_loss": policy_loss.item(), # Last batch loss
                    "value_loss": value_loss.item(),   # Last batch loss
                    "total_loss": total_loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                }
                if (iter + 1) % SL_TEST_INTERVAL == 0:
                    wandb_log["test_policy_loss"] = test_policy_loss
                    wandb_log["test_value_loss"] = test_value_loss
                wandb.log(wandb_log)

            iter += 1
            scheduler.step()
        
        avg_policy_loss = epoch_policy_loss / total_batches
        avg_value_loss = epoch_value_loss / total_batches
        
        print(f'Epoch {epoch + 1} Finished | Avg Train Policy: {avg_policy_loss:.4f} | Avg Train Value: {avg_value_loss:.4f}')

        # Save Checkpoint
        if (epoch + 1) % SL_CHECKPOINT_INTERVAL == 0:
            save_checkpoint(epoch, model.state_dict(), optimizer.state_dict(), scheduler.state_dict())
            print(f"Checkpoint saved at epoch {epoch + 1}")
            
        # Save Best Model (Overwrite)
        torch.save({'model_state_dict': model.state_dict()}, SL_MODEL_PATH)

if __name__ == "__main__":
    train_from_records()
