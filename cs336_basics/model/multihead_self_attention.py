import einops
from jaxtyping import Float, Int
import torch
import torch.nn as nn
from torch import Tensor

from cs336_basics.functions import scaled_dot_product_attention
from cs336_basics.model.linear import Linear
from cs336_basics.model.rope import RoPE

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None=None,
        max_seq_len: int | None=None,
        device=None,
        dtype=None
    ):
        super().__init__()
        assert d_model % num_heads == 0, f"d_model: {d_model} and num_heads: {num_heads}"
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, dtype=dtype, device=device)
        self.k_proj = Linear(d_model, d_model, dtype=dtype, device=device)
        self.v_proj = Linear(d_model, d_model, dtype=dtype, device=device)
        self.o_proj = Linear(d_model, d_model, dtype=dtype, device=device)
        self.rope = None
        if theta and max_seq_len:
            self.rope = RoPE(theta, self.d_k, max_seq_len, device)

    def forward(
        self,
        in_features: Float[Tensor, " ... seq_len d_model"],
        token_positions: Int[torch.Tensor, "... seq_len"] | None=None,
    ) -> torch.Tensor:
        seq_len = in_features.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)).bool()
        Q = einops.rearrange(self.q_proj(in_features), "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        K = einops.rearrange(self.k_proj(in_features), "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        V = einops.rearrange(self.v_proj(in_features), "... seq (head d_v) -> ... head seq d_v", head=self.num_heads)
        if token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        attention = scaled_dot_product_attention(Q, K, V, mask)
        return self.o_proj(einops.rearrange(attention, "... head seq d_v -> ... seq (head d_v)"))
