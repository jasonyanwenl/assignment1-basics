from jaxtyping import Float
import torch
import torch.nn as nn
from torch import Tensor

from cs336_basics.model.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.rope import RoPE
from cs336_basics.model.silu_ff import SiLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, device, dtype)
        self.ffn = SiLU(d_model, d_ff, device, dtype)

    def forward(self, x: Float[Tensor, "... seq d_model"], rope: RoPE | None=None) -> Float[Tensor, "... seq d_model"]:
        token_positions = torch.arange(x.shape[-2], device=x.device, dtype=torch.long)
        hidden = x + self.attn(self.ln1(x), rope, token_positions)
        return hidden + self.ffn(self.ln2(hidden))
