import torch

class RandomDataset(torch.utils.data.Dataset):
    def __init__(self,
                 len : int,
                 input_shape : tuple,
                 output_shape : tuple) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.len = len
        
    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x = torch.rand(self.input_shape)
        y = torch.rand(self.output_shape)
        return x, y