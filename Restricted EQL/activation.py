import torch
import torch.nn as nn

class PowerActivation(nn.Module):
    # input_dim: int, the dimension of the input
    #Let the input be a let x be a vector x = (x1,x2,x3,x4....) 
    # and this activation function f(x) = x1^w1*x2^w2*x3^w3*x4^w4....
    # we assume x is non-negative 
    def __init__(self, input_dim):
        super(PowerActivation, self).__init__()
        # Initialize the weights and scalar with 1
        self.w = nn.Parameter(torch.ones(input_dim)) 
        self.C = nn.Parameter(torch.tensor(1.0))
        
    def forward(self, x):
        #x = x + 1e-8  
        return self.C * torch.prod(x ** self.w, dim=-1, keepdim=True)
