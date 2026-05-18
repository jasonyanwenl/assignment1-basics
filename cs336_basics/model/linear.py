from math import sqrt
from einops import einsum
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                tensor=torch.empty(out_features, in_features, dtype=dtype, device=device),
                mean=0,
                std=std,
                a=-3 * std,
                b=3 * std
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
