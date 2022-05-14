import sys
import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

import torch
import torchvision
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data import CIFARDataModule

def train(model, model_name="test_1", data_dir="./data"):
    data_module = CIFARDataModule(batch_size=64, data_dir=data_dir)
    wandb_logger = WandbLogger(project='transplanter-cnn', job_type='train')

    # Initialize a trainer
    trainer = pl.Trainer(max_epochs=40,
                        progress_bar_refresh_rate=20, 
                        gpus=1, 
                        logger=wandb_logger)

    trainer.fit(model, data_module)
    model_path = os.path.join("trained_models", model_name + ".ckpt")
    trainer.save_checkpoint(model_path)
    trainer.test()
    wandb.finish()

if __name__ == "__main__":
    from models import SmallConvNet, BigConvNet
    # model = SmallConvNet(hidden_dim=64, flattened_size=512)
    model = BigConvNet(hidden_dim=128, flattened_size=2048)
    model_name = "hd_100"
    train(model=model, model_name=model_name)