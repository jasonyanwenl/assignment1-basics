from jaxtyping import Int
import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                tensor=torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device),
                mean=0,
                std=1,
                a=-3,
                b=3
            )
        )

    def forward(self, token_ids: Int[torch.Tensor, "... seq"]) -> torch.Tensor:
        return self.weight[token_ids]
