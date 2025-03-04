# simple_cnn.py
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    input:[batch_size, 3] (R, C, X_t)
    output:[batch_size, 1] (X_{t+1})
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        self.preprocess = nn.Sequential(
            nn.Linear(3, 8), 
            nn.ReLU()
        )
        
        # 1D conv
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1), 
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.fc = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        # input shape: [batch, 3]
        
        x = self.preprocess(x)          # [batch, 8]
        x = x.unsqueeze(1)              # [batch, 1, 8]
        
        x = self.conv_layers(x)         # [batch, 32, 1]
        x = x.view(x.size(0), -1)       # [batch, 32]
        
        return self.fc(x)             # [batch, 1]
