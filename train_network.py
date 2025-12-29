import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from network import DN_INPUT_SHAPE, load_model, DualNetwork
from hparams import rn_epochs, batch_size
import numpy as np
import pickle


def load_data():
    history_path = sorted(Path('./data').glob('*.history'))[-1]
    with history_path.open(mode='rb') as f:
        return pickle.load(f)


def train_network():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    history = load_data()
    xs, y_policies, y_values = zip(*history)

    # PyTorch uses (N, C, H, W) format
    c, a, b = DN_INPUT_SHAPE
    xs = np.array(xs)
    xs = xs.reshape(len(xs), c, a, b)  # Already in correct format for PyTorch
    y_policies = np.array(y_policies)
    y_values = np.array(y_values).reshape(-1, 1)  # Reshape for MSE loss

    # Convert to PyTorch tensors
    xs_tensor = torch.FloatTensor(xs)
    y_policies_tensor = torch.FloatTensor(y_policies)
    y_values_tensor = torch.FloatTensor(y_values)
    
    # Create dataset and dataloader
    dataset = TensorDataset(xs_tensor, y_policies_tensor, y_values_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load model
    model = load_model('./model/best.pth')
    model = model.to(device)
    model.train()

    # Define loss functions
    policy_criterion = nn.CrossEntropyLoss()  # For categorical cross-entropy
    value_criterion = nn.MSELoss()

    # Define optimizer with weight decay (L2 regularization)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0005)

    # Learning rate scheduler
    def lr_lambda(epoch):
        if epoch >= 80:
            return 0.25
        elif epoch >= 50:
            return 0.5
        else:
            return 1.0
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Training loop
    for epoch in range(rn_epochs):
        print(f'\rTraining network {epoch + 1}/{rn_epochs}', end='')
        
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        num_batches = 0
        
        for batch_x, batch_y_policy, batch_y_value in dataloader:
            batch_x = batch_x.to(device)
            batch_y_policy = batch_y_policy.to(device)
            batch_y_value = batch_y_value.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_policy, pred_value = model(batch_x)
            
            # Calculate losses
            # For policy, use KL divergence or cross-entropy with soft labels
            policy_loss = -torch.sum(batch_y_policy * torch.log(pred_policy + 1e-8)) / batch_x.size(0)
            value_loss = value_criterion(pred_value, batch_y_value)
            
            # Combined loss
            total_loss = policy_loss + value_loss
            
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            epoch_policy_loss += policy_loss.item()
            epoch_value_loss += value_loss.item()
            num_batches += 1
        
        scheduler.step()
        
    print('')

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
    }, './model/latest.pth')
    
    # Move model back to CPU and cleanup
    model = model.cpu()
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
