import torch
import torch.nn as nn

class ValueModel(nn.Module):
    '''
        A value model which maps the input embedding vector 
        and maps it to a reward value between 0 and 1
    '''
    def __init__(self, input_dim: int, dropout: float = 0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=1),
        )

    def forward(self, x: torch.tensor):
        logits = self.mlp(x) # shape: (batch, 1)
        return logits