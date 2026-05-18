from jaxtyping import Float, Int
from torch import nn
import torch


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device:torch.device | None = None):
        super().__init__()
        assert d_k % 2 == 0, f"d_k: {d_k} is not an even number!"
        freqs_inv = 1.0 / (theta ** ((2 * torch.arange(1, d_k // 2 + 1) - 2) / d_k))
        positions = torch.arange(max_seq_len)
        angles = torch.outer(positions, freqs_inv)
        self.register_buffer("cos", angles.cos().to(device), persistent=False)
        self.register_buffer("sin", angles.sin().to(device), persistent=False)

    def forward(self, x: Float[torch.Tensor, "... seq_len d_k"], token_positions: Int[torch.Tensor, "... seq_len"]) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        even = x_even * cos - x_odd * sin
        odd = x_even * sin + x_odd * cos
        return torch.stack([even, odd], dim=-1).flatten(start_dim=-2)