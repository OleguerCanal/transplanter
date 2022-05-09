import os
import sys
import os

import torch
from torch.nn import functional as F
import torch.nn as nn
import pytorch_lightning as pl

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules import LinearBlock, ConvBlock

## LinearNet
class LinearNet(torch.nn.Module):
    def __init__(self,
                 in_features : int,
                 out_features : int,
                 n_blocks : int) -> None:
        super().__init__()

        self.hidden_layers = torch.nn.ModuleList(
            modules = [LinearBlock(in_features, in_features) for _ in range(n_blocks)]
        )
        self.out_layer = torch.nn.Linear(in_features, out_features)


    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = layer(x)
        return self.out_layer(x)


class ConvNet(pl.LightningModule):
    def __init__(self,
                 in_features : int,
                 hidden_dim: int,
                 out_features : int,
                 n_blocks : int) -> None:
        super().__init__()

        modules = [ConvBlock(in_features, hidden_dim, pool_size=2)]
        modules += [ConvBlock(hidden_dim, hidden_dim, pool_size=2)]
        modules += [ConvBlock(hidden_dim, hidden_dim) for _ in range(n_blocks-2)]
        self.hidden_layers = torch.nn.ModuleList(modules = modules)
        self.out_layer_1 = torch.nn.Linear(1024, 250)
        self.out_layer_2 = torch.nn.Linear(250, out_features)
        self.dropout = torch.nn.Dropout(p=0.2)
        self.activation = torch.nn.ReLU()
        self.out_features = out_features

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """ Input: (B,C,H,W)
        """
        for layer in self.hidden_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.activation(self.out_layer_1(x)))
        return self.out_layer_2(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1.5e-4)
        return optimizer

    def _get_metrics(self, batch):
        x, y = batch
        logits = self.forward(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        accuracy = torch.sum(logits.argmax(dim=1) == y)/x.size(0)
        return loss, accuracy

    def training_step(self, train_batch, batch_idx):
        loss, accuracy = self._get_metrics(train_batch)
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss, accuracy = self._get_metrics(val_batch)
        self.log('val_loss', loss)
        self.log('val_accuracy', accuracy)
        return loss

    def test_step(self, test_batch):
        loss, accuracy = self._get_metrics(test_batch)
        self.log('test_loss', loss)
        self.log('test_accuracy', accuracy)
        return loss