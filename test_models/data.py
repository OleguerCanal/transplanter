from typing import Optional
import pytorch_lightning as pl
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split


class CIFARDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # Train/val
        cifar_full = CIFAR10(root=self.data_dir, train=True, download=True, transform=transform)
        n_train, n_val = int(len(cifar_full)*0.9), int(len(cifar_full)*0.1)
        self.cifar_train, self.cifar_val = random_split(cifar_full, [n_train, n_val])
        
        # Test
        self.cifar_test = CIFAR10(self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.cifar_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.cifar_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.cifar_test, batch_size=self.batch_size)
