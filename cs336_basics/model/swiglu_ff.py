import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None, device=None, dtype=None):
        super().__init__()
        assert d_model >= 64, f"d_model: {d_model} too small"
        d_ff_rounded = d_ff if d_ff else (8 * d_model) // (3 * 64) * 64
        self.w1_weight = nn.Parameter(
            nn.init.trunc_normal_(
                tensor=torch.empty(d_ff_rounded, d_model, dtype=dtype, device=device),
                mean=0,
                std=1,
                a=-3,
                b=3
            )
        )
        self.w2_weight = nn.Parameter(
            nn.init.trunc_normal_(
                tensor=torch.empty(d_model, d_ff_rounded, dtype=dtype, device=device),
                mean=0,
                std=1,
                a=-3,
                b=3
            )
        )
        self.w3_weight = nn.Parameter(
            nn.init.trunc_normal_(
                tensor=torch.empty(d_ff_rounded, d_model, dtype=dtype, device=device),
                mean=0,
                std=1,
                a=-3,
                b=3
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self._silu(x @ self.w1_weight.T) * (x @ self.w3_weight.T)) @ self.w2_weight.T

    def _silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
