import torch
import torchmetrics
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.datasets as datasets
from timm.data.mixup import Mixup
from torch_optimizer import MADGRAD
from model.patch_conv_net import PatchConvNet
from dataset.transform import train_transform, test_transform


class Cifar100Model(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.patch_conv_net = PatchConvNet(cfg)
        self.criterion = nn.CrossEntropyLoss()
        self.val_acc = torchmetrics.Accuracy()
        self.mixup_fn = Mixup(**cfg.mixup_args)
        self.lr = cfg.lr
    
    def forward(self, x, training):
        return self.patch_conv_net(x, training)

    def shared_step(self, batch, batch_idx, training):
        images, target = batch
        if training and self.cfg.use_mixup:
            images, target = self.mixup_fn(images, target)
        training = training and self.cfg.use_stochastic_depth
        output = self(images, training)
        loss = self.criterion(output, target)
        return loss, output, target
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self.shared_step(batch, batch_idx, True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, output, target = self.shared_step(batch, batch_idx, False)
        self.val_acc(output, target)
        logs = {'val_loss': loss, 'val_acc': self.val_acc}
        self.log_dict(logs, prog_bar=True)

    def configure_optimizers(self):
        optimizer = MADGRAD(self.parameters(), lr=self.lr)
        return optimizer
    
    def train_dataloader(self):
        train_dataset = datasets.CIFAR100(self.cfg.data_dir, train=True, download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.cfg.train_batch_size, shuffle=True, num_workers=self.cfg.num_workers
        )
        return train_loader

    def val_dataloader(self):
        val_dataset = datasets.CIFAR100(self.cfg.data_dir, train=False, download=True, transform=test_transform)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=self.cfg.train_batch_size, shuffle=False, num_workers=self.cfg.num_workers
        )
        return val_loader