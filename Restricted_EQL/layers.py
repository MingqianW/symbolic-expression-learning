import torch
import torch.nn as nn
import torch.nn.functional as F

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
    

class ChoiceLayerSTE(nn.Module):
    def __init__(self, num_inputs=2):
        super().__init__()
        self.logits = nn.Parameter(torch.randn(num_inputs))  # Learnable scores

    def forward(self, *inputs):
        probs = F.softmax(self.logits, dim=0)                # [num_inputs]
        idx = torch.argmax(probs)
        hard_mask = torch.zeros_like(probs)
        hard_mask[idx] = 1.0
        mask = (hard_mask - probs).detach() + probs          # STE: forward is hard(one-hot), backward is soft
        out = sum(m * inp for m, inp in zip(mask, inputs))   # shape of inputs[0]
        return out

    def selected(self):
        # Returns the index of the currently selected input (for interpretability)
        return torch.argmax(self.logits).item()
