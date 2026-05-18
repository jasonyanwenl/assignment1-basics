from einops import einsum
from jaxtyping import Float
import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[torch.Tensor, "... seq d_model"]) -> Float[torch.Tensor, "... seq d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt(einsum(x, x, "... d_model, ... d_model -> ...") / self.d_model + self.eps)
        result = einsum(x, 1 / rms, "... d_model, ... -> ... d_model") * self.weight

        return result.to(in_dtype)