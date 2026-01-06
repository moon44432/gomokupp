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

def save_checkpoint(epoch, model_state, optimizer_state):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model_state,
        'optimizer_state_dict': optimizer_state,
    }, SL_MODEL_PATH + ".checkpoint")
    
    with open(SL_CHECKPOINT_FILE, 'w') as f:
        json.dump({'epoch': epoch}, f)

def load_checkpoint():
    if os.path.exists(SL_CHECKPOINT_FILE):
        with open(SL_CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

def ensure_dataset_exists():
    if dataset_dir.exists() and any(dataset_dir.glob('*.pkl')):
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
    
    for i in range(0, len(history), SL_CHUNK_SIZE):
        chunk = history[i:i + SL_CHUNK_SIZE]
        print(f"Processing chunk {i // SL_CHUNK_SIZE + 1} / {(len(history) + SL_CHUNK_SIZE - 1) // SL_CHUNK_SIZE}...")
            
        output_path = dataset_dir / f"chunk_{i // SL_CHUNK_SIZE:03d}.pkl"
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

def train_from_records():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name="train_supervised")

    # Prepare Data
    ensure_dataset_exists()
    dataset_files = list(dataset_dir.glob('*.pkl'))
    print(f"Found {len(dataset_files)} dataset files.")

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
    optimizer = optim.SGD(model.parameters(), lr=SL_LEARNING_RATE, momentum=0.9, weight_decay=0.0001)
    value_criterion = nn.MSELoss()

    # Resume
    start_epoch = 0
    checkpoint_info = load_checkpoint()
    if checkpoint_info:
        start_epoch = checkpoint_info['epoch'] + 1
        if os.path.exists(SL_MODEL_PATH + ".checkpoint"):
            checkpoint = torch.load(SL_MODEL_PATH + ".checkpoint")
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Resuming from epoch {start_epoch}")

    # Train Loop
    for epoch in range(start_epoch, SL_EPOCHS):
        print(f"Epoch {epoch + 1}/{SL_EPOCHS}")
        random.shuffle(dataset_files)
        
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        total_batches = 0
        
        for file_idx, file_path in enumerate(dataset_files):
            print(f"  Loading {file_path.name} ({file_idx + 1}/{len(dataset_files)})...")
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

                if USE_WANDB:
                    wandb.log({
                        "epoch": epoch,
                        "file": file_path.name,
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "total_loss": total_loss.item()
                    })
        
        avg_policy_loss = epoch_policy_loss / total_batches
        avg_value_loss = epoch_value_loss / total_batches
        
        print(f'Epoch {epoch + 1} Finished | Policy Loss: {avg_policy_loss:.4f} | Value Loss: {avg_value_loss:.4f}')

        # Save Checkpoint
        if (epoch + 1) % SL_CHECKPOINT_INTERVAL == 0:
            save_checkpoint(epoch, model.state_dict(), optimizer.state_dict())
            print(f"Checkpoint saved at epoch {epoch + 1}")
            
        # Save Best Model (Overwrite)
        torch.save({'model_state_dict': model.state_dict()}, SL_MODEL_PATH)

if __name__ == "__main__":
    train_from_records()
