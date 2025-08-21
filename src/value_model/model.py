import torch
import torch.nn as nn

class ValueModel(nn.Module):
    '''
        A value model which maps the input embedding vector 
        and maps it to a reward value between 0 and 1
    '''
    def __init__(self, input_dim: int, hidden_dims: list, dropout: float = 0):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        if hidden_dims != []:
        # Build hidden layers with dropout
            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim
        
        # Output layer (maps to single value)
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.tensor):
        logits = self.mlp(x)  # shape: (batch, 1)
        return logits