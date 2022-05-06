import torch

# from attentions.attentions import MultiHeadAttention


## LinearNet
class LinearBlock(torch.nn.Module):
    def __init__(self,
                 in_features : int,
                 out_features : int) -> None:
        super().__init__()
        self.layer_1 = torch.nn.Linear(in_features, out_features)
        # self.layer_2 = torch.nn.Linear(out_features, out_features)
        self.activation = torch.nn.LeakyReLU(0.01)

    def forward(self,
                x : torch.Tensor) -> torch.Tensor:
        x = self.activation(self.layer_1(x))
        # return self.activation(self.layer_2(x))
        return x

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


## ConvNet
class ConvBlock(torch.nn.Module):
    def __init__(self, 
                 in_channels : int,
                 out_channels : int,
                 kernel_size : int = 3) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.activation = torch.nn.ReLU()
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.pool(self.activation(self.conv(x)))


class ConvNet(torch.nn.Module):
    def __init__(self,
                 in_features : int,
                 hidden_dim: int,
                 out_features : int,
                 n_blocks : int) -> None:
        super().__init__()

        modules = [ConvBlock(in_features, hidden_dim)]
        modules += [ConvBlock(hidden_dim, hidden_dim) for _ in range(n_blocks-1)]
        self.hidden_layers = torch.nn.ModuleList(modules = modules)
        self.out_layer = torch.nn.Linear(256, out_features)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """ Input: (B,C,H,W)
        """
        for layer in self.hidden_layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        return self.out_layer(x)