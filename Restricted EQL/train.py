import torch
import torch.nn as nn
from inspect import signature

DOMAIN = (1, 2)    # Domain of dataset - range from which we sample x
# DOMAIN = np.array([[0, -1, -1], [1, 1, 1]])  # Use this format if each input variable has a different domain

def generate_data(func, N, range_min=DOMAIN[0], range_max=DOMAIN[1]):
    """Generates datasets."""
    x_dim = len(signature(func).parameters)     # Number of inputs to the function, or, dimensionality of x
    x = (range_max - range_min) * torch.rand([N, x_dim]) + range_min
    y = torch.tensor([[func(*x_i)] for x_i in x])
    return x, y #x.shape: [N, x_dim] and y.shape: [N, 1]

