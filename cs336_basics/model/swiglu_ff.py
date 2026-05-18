from jaxtyping import Float
import torch
import torch.nn as nn

from cs336_basics.model.linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None, device=None, dtype=None):
        super().__init__()
        assert d_model >= 64, f"d_model: {d_model} too small"
        d_ff_rounded = d_ff if d_ff else (8 * d_model) // (3 * 64) * 64
        self.w1 = Linear(d_model, d_ff_rounded, dtype=dtype, device=device)
        self.w2 = Linear(d_ff_rounded, d_model, dtype=dtype, device=device)
        self.w3 = Linear(d_model, d_ff_rounded, dtype=dtype, device=device)

    def forward(self, x: Float[torch.Tensor, "... seq d_model"]) -> Float[torch.Tensor, "... seq d_model"]:
        return self.w2((self._silu(self.w1(x)) * (self.w3(x))))

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
