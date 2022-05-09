import torch

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self,
                 len : int,
                 input_shape : tuple) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.len = len
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = torch.rand(self.input_shape)
        return x, torch.empty(0)