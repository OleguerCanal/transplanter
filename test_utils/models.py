from turtle import forward
import torch

# from attentions.attentions import MultiHeadAttention

class LinearBlock(torch.nn.Module):
    def __init__(self,
                 in_features : int,
                 out_features : int) -> None:
        super().__init__()
        self.layer_1 = torch.nn.Linear(in_features, out_features)
        self.activation = torch.nn.LeakyReLU(0.01)

    def forward(self,
                x : torch.Tensor) -> torch.Tensor:
        return self.activation(self.layer_1(x))


class Net(torch.nn.Module):
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