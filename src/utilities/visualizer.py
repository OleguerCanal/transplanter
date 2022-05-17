import torch
import numpy as np

import matplotlib.pyplot as plt

def visualize_tensor(tensor : torch.Tensor):
    tensor = tensor.detach().cpu().numpy()
    dims = len(tensor.shape)
    print("visualizing", dims)
    if dims == 2:
        plt.imshow(tensor)
        plt.show()