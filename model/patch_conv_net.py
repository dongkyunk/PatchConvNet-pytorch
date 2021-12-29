import torch
import torch.nn as nn
from torch.nn.modules import Conv2d
from model.custom_blocks import Conv2dBlock, SEBlock
from torchvision.ops import stochastic_depth


class ConvolutionalStem(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims, num_layers):
        super().__init__()
        assert len(hidden_dims) + 1 == num_layers 
        dims = [in_dim] + hidden_dims + [out_dim]

        self.layers = [Conv2dBlock(dims[i], dims[i+1], 3, 2, 1, 'gelu') for i in range(num_layers)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class ColumnBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim=None):
        super().__init__()
        if not hidden_dim:
            hidden_dim = in_dim // 4

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            Conv2dBlock(in_dim, hidden_dim, 1, 1, 0, 'gelu'),
            Conv2dBlock(hidden_dim, hidden_dim, 3, 1, 1, 'gelu'),
            SEBlock(hidden_dim),
            Conv2d(hidden_dim, in_dim, 1, 1, 0),
        )

    def forward(self, x, training):
        x_skip = x
        x = self.layers(x)
        x = stochastic_depth(x, p=0.5, mode="row", training=training)
        x = x + x_skip
        return x


class Column(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, num_layers=60):
        super().__init__()
        self.layers = [ColumnBlock(in_dim, hidden_dim) for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.layers)
    
    def forward(self, x, training):
        for layer in self.layers:
            x = layer(x, training)
        return x

    
class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.cls_vec = nn.Parameter(torch.randn(in_dim))
        self.fc = nn.Linear(in_dim, in_dim)
        self.softmax = nn.Softmax(-1)

    def forward(self, x):
        weights = torch.matmul(x.view(-1, x.shape[1]), self.cls_vec)
        weights = self.softmax(weights.view(x.shape[0], -1))
        x = torch.bmm(x.view(x.shape[0], x.shape[1], -1), weights.unsqueeze(-1)).squeeze()
        x = x + self.cls_vec
        x = self.fc(x)
        x = x + self.cls_vec
        return x


class PatchConvNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.conv_stem = ConvolutionalStem(
            in_dim=cfg.input_dim, 
            out_dim=cfg.patch_dim, 
            hidden_dims=cfg.conv_stem_hidden_dims, 
            num_layers=cfg.conv_stem_layers
        )
        self.column = Column(
            in_dim=cfg.patch_dim, 
            hidden_dim=cfg.column_hidden_dim, 
            num_layers=cfg.model_depth
        )
        self.pooling = AttentionPooling(
            in_dim=cfg.patch_dim, 
        )
        self.head = nn.Linear(cfg.patch_dim, cfg.num_of_classes)

    def forward(self, x, training):
        x = self.conv_stem(x)
        x = self.column(x, training)
        x = self.pooling(x)
        x = self.head(x)
        return x

