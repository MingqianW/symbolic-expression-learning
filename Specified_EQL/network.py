import torch
import torch.nn as nn
from Restricted_EQL.layers import PowerActivation 

class SpecifiedSctureModel1(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        # Term 1: nested power of sums
        self.linear_sum = nn.Linear(input_dim, input_dim, bias=False)  
        with torch.no_grad():
            nn.init.uniform_(self.linear_sum.weight, a=0.0, b=1.5)    
        self.power_sum = PowerActivation(input_dim)             

        # Term 2: linear combination
        self.linear_term = nn.Linear(input_dim, 1, bias=False)                    
        with torch.no_grad():
            nn.init.uniform_(self.linear_term.weight, a=0.0, b=1.5)
        # Term 3: power product directly on inputs
        self.power_direct = PowerActivation(input_dim)                     
        
    def forward(self, x):
        # Term 1
        x_sum = self.linear_sum(x)                 # shape [N, input_dim]
        term1 = self.power_sum(x_sum)              # shape [N, 1]

        # Term 2
        term2 = self.linear_term(x)                # shape [N, 1]

        # Term 3
        term3 = self.power_direct(x)      # shape [N, 1]

        return term1 + term2 + term3

class SpecifiedSctureModel2(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim

        # Term 1: nested power of sums
        self.linear_sum = nn.Linear(input_dim, input_dim, bias=False)      
        self.power_sum = PowerActivation(input_dim)             
        with torch.no_grad():
            nn.init.uniform_(self.linear_sum.weight, a=0.0, b=1.5)
        # Term 3: power product on raw inputs
        self.power_direct = PowerActivation(input_dim)          
        
        # Final linear over [term1, term3, x1, ..., xd]
        self.final_linear = nn.Linear(2 + input_dim, 1, bias=False) 
        with torch.no_grad():
            nn.init.uniform_(self.final_linear.weight, a=0.0, b=1.5) 
        
    def forward(self, x):
        term1 = self.power_sum(self.linear_sum(x))  # shape [N, 1]
        
        term3 = self.power_direct(x)                # shape [N, 1]
        
        combined = torch.cat([term1, term3, x], dim=1)  # shape [N, 2 + input_dim]
        
        return self.final_linear(combined)               # shape [N, 1]