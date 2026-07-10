from jaxtyping import Float
import torch
import torch.nn as nn

from cs336_basics.model.linear import Linear

class SiLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None, device=None, dtype=None):
        super().__init__()
        d_ff_rounded = d_ff if d_ff else 4 * d_model
        self.w1 = Linear(d_model, d_ff_rounded, dtype=dtype, device=device)
        self.w2 = Linear(d_ff_rounded, d_model, dtype=dtype, device=device)

    def forward(self, x: Float[torch.Tensor, "... seq d_model"]) -> Float[torch.Tensor, "... seq d_model"]:
        return self.w2(self.silu(self.w1(x)))

    @classmethod
    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
