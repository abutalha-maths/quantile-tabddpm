import torch
import torch.nn as nn
from typing import List

class MLPDiffusion(nn.Module):
    """Enhanced MLP model for TabDDPM with deeper architecture and batch normalization."""
    def __init__(self, d_in: int, d_layers: List[int], d_out: int, dropout: float = 0.3):
        super(MLPDiffusion, self).__init__()
        layers = []
        prev_dim = d_in
        for dim in d_layers:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, d_out))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)