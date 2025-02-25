import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomNN(nn.Module):
    def __init__(self, n, hidden_dim=64):
        """
        Neural network with:
        - n: Number of points (each consisting of X, R, C)
        - hidden_dim: Number of neurons in the hidden layer (default 64)
        Maps n X-values to a single neuron, n R-values to a single neuron, and n C-values to a single neuron.
        """
        super(CustomNN, self).__init__()

        self.n = n  # Number of (X, R, C) triplets
        self.X_layer = nn.Linear(n, hidden_dim)
        self.r_layer = nn.Linear(n, hidden_dim)
        self.c_layer = nn.Linear(n, hidden_dim)
        # 1D convolutional layer so that can interaction with each tuple of (X, R, C) and maybe provide some nonlinearity
        self.interaction_layer =  nn.Conv1d(in_channels=1, out_channels=hidden_dim, kernel_size=3, stride=3)

        #Final layer: Maps (hidden_dim * 4) to n outputs (Y1 to Yn)
        #I dont think the final layer design is good, if i only use linear layer, the model will be a linear model essentially
        self.output_layer = nn.Linear(hidden_dim * 4, n)

    def forward(self, input_tensor):
        """
        Forward pass:
        - Splits input into X, R, and C components
        - Maps each component through its respective layer
        - Concatenates results and maps to output Y
        """
        # Split input
        X = input_tensor[:, 0::3]
        r = input_tensor[:, 1::3]  
        c = input_tensor[:, 2::3]  

        X_out = F.relu(self.X_layer(X))
        r_out = F.relu(self.r_layer(r))
        c_out = F.relu(self.c_layer(c))
        input_reshaped = input_tensor.unsqueeze(1)  # (batch_size, 1, n * 3)
        interaction_out = F.relu(self.interaction_layer(input_reshaped))


        # Concatenate the outputs of X, R, and C neurons
        combined = torch.cat([X_out, r_out, c_out,interaction_out], dim=1)

        # Final mapping to Y1, Y2, ..., Yn

        output = self.output_layer(combined)

        return output  # Shape: (batch_size, n)
