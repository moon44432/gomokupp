import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time

from hparams import BOARD_WIDTH, DN_FILTERS, DN_KERNEL_SIZE, DN_BLOCK_NUM, PREV_STATE_COUNT

input_shape = (2 + 2 * PREV_STATE_COUNT + 1, BOARD_WIDTH, BOARD_WIDTH)  # PyTorch uses (C, H, W) format
output_size = BOARD_WIDTH ** 2


def _init_conv(conv: nn.Conv2d) -> None:
    nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')


def _init_linear(fc: nn.Linear) -> None:
    nn.init.kaiming_normal_(fc.weight, mode='fan_out', nonlinearity='relu')
    if fc.bias is not None:
        nn.init.zeros_(fc.bias)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        _init_conv(self.conv1)
        _init_conv(self.conv2)
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.ones_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + residual
        x = F.relu(x)
        return x


class DualNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # ResNet-style stem
        padding = DN_KERNEL_SIZE // 2
        self.stem_conv = nn.Conv2d(input_shape[0], DN_FILTERS, DN_KERNEL_SIZE, padding=padding, bias=False)
        self.stem_bn = nn.BatchNorm2d(DN_FILTERS)
        _init_conv(self.stem_conv)
        nn.init.ones_(self.stem_bn.weight)
        nn.init.zeros_(self.stem_bn.bias)

        # Residual tower
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(DN_FILTERS, DN_KERNEL_SIZE) for _ in range(DN_BLOCK_NUM)]
        )

        # Policy head: 1x1 conv -> BN -> ReLU -> FC to board^2
        self.policy_conv = nn.Conv2d(DN_FILTERS, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * BOARD_WIDTH * BOARD_WIDTH, output_size)

        # Value head: 1x1 conv -> BN -> ReLU -> FC(256) -> ReLU -> FC(1) -> tanh
        self.value_conv = nn.Conv2d(DN_FILTERS, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * BOARD_WIDTH * BOARD_WIDTH, 256)
        self.value_fc2 = nn.Linear(256, 1)

        _init_conv(self.policy_conv)
        nn.init.ones_(self.policy_bn.weight)
        nn.init.zeros_(self.policy_bn.bias)
        _init_linear(self.policy_fc)

        _init_conv(self.value_conv)
        nn.init.ones_(self.value_bn.weight)
        nn.init.zeros_(self.value_bn.bias)
        _init_linear(self.value_fc1)
        _init_linear(self.value_fc2)
    
    def forward(self, x):
        x = self.stem_conv(x)
        x = self.stem_bn(x)
        x = F.relu(x)

        x = self.res_blocks(x)

        # Policy
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p)
        p = p.view(p.size(0), -1)
        p = self.policy_fc(p)
        p = F.softmax(p, dim=1)

        # Value
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v)
        v = v.view(v.size(0), -1)
        v = self.value_fc1(v)
        v = F.relu(v)
        v = self.value_fc2(v)
        v = torch.tanh(v)

        return p, v


def _try_load_state_dict(model: nn.Module, path: str) -> bool:
    try:
        checkpoint = torch.load(path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        return True
    except Exception:
        return False


def dual_network():
    """Initialize model if it doesn't exist"""
    os.makedirs('./model/', exist_ok=True)

    best_path = './model/best.pth'
    if os.path.exists(best_path):
        # If the architecture changed and the checkpoint is incompatible,
        # back it up and re-initialize.
        model = DualNetwork()
        if _try_load_state_dict(model, best_path):
            return
        backup_path = f"./model/best.incompatible.{int(time.time())}.pth"
        try:
            os.replace(best_path, backup_path)
            print(f"Existing checkpoint incompatible with current network. Backed up to: {backup_path}")
        except Exception:
            # If backup fails, we still proceed to overwrite best.pth.
            print("Existing checkpoint incompatible with current network. Overwriting best.pth")

    model = DualNetwork()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    torch.save({'model_state_dict': model.state_dict()}, best_path)
    del model


def load_model(path):
    """Load a PyTorch model from path"""
    model = DualNetwork()
    checkpoint = torch.load(path, map_location='cpu')
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load model checkpoint '{path}'. "
            f"Original error: {e}"
        )
    model.eval()
    return model
