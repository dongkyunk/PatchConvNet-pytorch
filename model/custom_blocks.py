import torch
import torch.nn as nn


def get_activation(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'gelu':
        return nn.GELU()
    elif act is None:
        return nn.Identity()


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, act=None):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding)
        self.act = get_activation(act)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_dim, reduction=16):
        super().__init__()
        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(),
            nn.Linear(in_dim, in_dim//reduction, bias=False),
            nn.ReLU(), 
            nn.Linear(in_dim//reduction, in_dim, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        weights = self.layers(x)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        return x * weights.expand_as(x)


