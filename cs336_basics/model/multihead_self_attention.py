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
        self.output_proj = Linear(d_model, d_model, dtype=dtype, device=device)

    def forward(
        self,
        in_features: Float[Tensor, " ... seq_len d_model"],
        rope: RoPE | None=None,
        token_positions: Int[torch.Tensor, "... seq_len"] | None=None,
    ) -> Float[Tensor, "... seq_len d_model"]:
        seq_len = in_features.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=in_features.device)).bool()
        Q = einops.rearrange(self.q_proj(in_features), "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        K = einops.rearrange(self.k_proj(in_features), "... seq (head d_k) -> ... head seq d_k", head=self.num_heads)
        V = einops.rearrange(self.v_proj(in_features), "... seq (head d_v) -> ... head seq d_v", head=self.num_heads)
        if rope is not None and token_positions is not None:
            Q = rope(Q, token_positions)
            K = rope(K, token_positions)
        attention = scaled_dot_product_attention(Q, K, V, mask)
        return self.output_proj(einops.rearrange(attention, "... head seq d_v -> ... seq (head d_v)"))
