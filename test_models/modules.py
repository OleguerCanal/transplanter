import torch

class LinearBlock(torch.nn.Module):
    def __init__(self,
                 in_features : int,
                 out_features : int) -> None:
        super().__init__()
        self.layer_1 = torch.nn.Linear(in_features, out_features)
        self.layer_2 = torch.nn.Linear(out_features, out_features)
        self.activation = torch.nn.LeakyReLU(0.01)
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self,
                x : torch.Tensor) -> torch.Tensor:
        x = self.dropout(self.activation(self.layer_1(x)))
        return self.activation(self.layer_2(x))

## ConvNet
class ConvBlock(torch.nn.Module):
    def __init__(self, 
                 in_channels : int,
                 out_channels : int,
                 kernel_size : int = 3,
                 pool_size : int = None) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool = None if pool_size is None else torch.nn.MaxPool2d(kernel_size=pool_size)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.2)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.pool is not None:
            x = self.pool(self.activation(self.conv(x)))
        else:
            x = self.activation(self.conv(x))
        return self.dropout(x)