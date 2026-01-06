import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import os
import json
import pickle
import random
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from pathlib import Path

from network import input_shape, load_model, dual_network
from self_play import self_play
from evaluate import evaluate_network
from rule import Renju
from hparams import *


def save_checkpoint(cycle_idx):
    with open(RL_CHECKPOINT_FILE, 'w') as f:
        json.dump({'cycle_idx': cycle_idx}, f)

def load_checkpoint():
    if os.path.exists(RL_CHECKPOINT_FILE):
        with open(RL_CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return None

def load_latest_sp_data():
    sp_history_path = sorted(Path('./data').glob('*_sp.history'))[-1]
    print(f"Loading data from {sp_history_path}")
    with sp_history_path.open(mode='rb') as f:
        data = pickle.load(f)
    random.shuffle(data)
    return data

def train_step(model, dataset, device, cycle_idx):
    dataloader = DataLoader(dataset, batch_size=RL_BATCH_SIZE, shuffle=True)
    
    lr = RL_LEARNING_RATE
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
    value_criterion = nn.MSELoss()
    
    model.train()
    
    for epoch in range(RL_EPOCHS):
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y_policy, batch_y_value in enumerate(tqdm(dataloader)):
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
            num_batches += 1
            
        avg_policy_loss = epoch_policy_loss / num_batches
        avg_value_loss = epoch_value_loss / num_batches
        
        print(f'Cycle {cycle_idx} Epoch {epoch + 1}/{RL_EPOCHS} | Policy Loss: {avg_policy_loss:.4f} | Value Loss: {avg_value_loss:.4f}')
        
        if USE_WANDB:
            wandb.log({
                "cycle": cycle_idx,
                "epoch": epoch,
                "policy_loss": avg_policy_loss,
                "value_loss": avg_value_loss,
                "total_loss": avg_policy_loss + avg_value_loss,
                "learning_rate": lr
            })

def train_from_sp():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if USE_WANDB:
        wandb.init(project=WANDB_PROJECT, name="train_from_sp")
    
    # Ensure model exists
    if not os.path.exists('./model/best.pth'):
        dual_network()

    rule = Renju()

    cycle = 0
    checkpoint = load_checkpoint()
    if checkpoint:
        cycle = checkpoint['cycle_idx'] + 1
        print(f"Resuming from cycle {cycle}")

    while True:
        print('Training {:04d}'.format(cycle + 1))

        # 1. Self Play
        print('Generating data from self playing...')
        self_play(rule)
            
        # 2. Load Data
        history = load_latest_sp_data()
        xs, y_policies, y_values = zip(*history)
        c, a, b = input_shape
        xs = np.array(xs).reshape(len(xs), c, a, b)
        y_policies = np.array(y_policies)
        y_values = np.array(y_values).reshape(-1, 1)
        
        dataset = TensorDataset(
            torch.FloatTensor(xs), 
            torch.FloatTensor(y_policies), 
            torch.FloatTensor(y_values)
        )

        # 3. Train
        print('Training network...')
        model = load_model('./model/best.pth')
        model = model.to(device)
        
        train_step(model, dataset, device, cycle)
        
        # Save latest model
        torch.save({'model_state_dict': model.state_dict()}, './model/latest.pth')
        
        # 4. Evaluate
        print('Evaluating network...')
        # evaluate_network compares latest.pth vs best.pth and updates best.pth if latest wins
        evaluate_network(rule)

        save_checkpoint(cycle)
        cycle += 1

if __name__ == "__main__":
    train_from_sp()
