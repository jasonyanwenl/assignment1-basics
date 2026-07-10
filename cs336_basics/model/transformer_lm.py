from jaxtyping import Float, Int
import torch
import torch.nn as nn
from torch import Tensor

from cs336_basics.model.embedding import Embedding
from cs336_basics.model.linear import Linear
from cs336_basics.model.rmsnorm import RMSNorm
from cs336_basics.model.rope import RoPE
from cs336_basics.model.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype
        )
        self.rope = RoPE(
            theta=rope_theta,
            d_k=d_model // num_heads,
            max_seq_len=context_length,
            device=device
        )
        self.layers = nn.ModuleList([TransformerBlock(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            device=device,
            dtype=dtype
        ) for _ in range(num_layers)])
        # self.ln_final = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: Int[Tensor, "... seq"]) -> Float[torch.Tensor, "... seq vocab_size"]:
        hidden = self.token_embeddings(in_indices)
        for layer in self.layers:
            hidden = layer(hidden, self.rope)
        return self.lm_head(hidden)
