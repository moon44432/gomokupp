import torch
import torch.nn as nn
import torch.nn.functional as F
from hparams import board_width, dn_filters, dn_kernel_size, dn_block_num
import os

DN_INPUT_SHAPE = (2, board_width, board_width)  # PyTorch uses (C, H, W) format
DN_OUTPUT_SIZE = board_width ** 2


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # He initialization
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x


class DualNetwork(nn.Module):
    def __init__(self):
        super(DualNetwork, self).__init__()
        
        # Initial conv block
        self.conv_blocks = nn.ModuleList()
        self.conv_blocks.append(ConvBlock(2, dn_filters, dn_kernel_size))
        
        # Residual conv blocks
        for _ in range(dn_block_num - 1):
            self.conv_blocks.append(ConvBlock(dn_filters, dn_filters, dn_kernel_size))
        
        # Global average pooling is done in forward
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Policy head
        self.policy_head = nn.Linear(dn_filters, DN_OUTPUT_SIZE)
        
        # Value head
        self.value_head = nn.Linear(dn_filters, 1)
        
        # Weight decay will be applied via optimizer
        # Initialize policy and value heads
        nn.init.kaiming_normal_(self.policy_head.weight)
        nn.init.kaiming_normal_(self.value_head.weight)
    
    def forward(self, x):
        # Forward through conv blocks
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        # Global average pooling
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        
        # Policy output
        p = self.policy_head(x)
        p = F.softmax(p, dim=1)
        
        # Value output
        v = self.value_head(x)
        v = torch.tanh(v)
        
        return p, v


def dual_network():
    """Initialize model if it doesn't exist"""
    if os.path.exists('./model/best.pth'):
        return
    
    model = DualNetwork()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    os.makedirs('./model/', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
    }, './model/best.pth')
    
    del model


def load_model(path):
    """Load a PyTorch model from path"""
    model = DualNetwork()
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
