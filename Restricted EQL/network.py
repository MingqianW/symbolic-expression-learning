import torch
import torch.nn as nn
import activation 

class MixedModel(nn.Module):
    def __init__(self, input_dim):
        super(MixedModel, self).__init__()
        self.power_activation = activation.PowerActivation(input_dim)
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        power_out = self.power_activation(x)  # [N, 1]
        linear_out = self.linear(x)           # [N, 1]
        return power_out + linear_out  # Should I just add them?