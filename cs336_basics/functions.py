import math
from einops import einsum
from jaxtyping import Float, Bool, Int
import torch
from torch import Tensor


def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, " ..."]:
    exp = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, "batch_size ... seq_len d_k"],
    K: Float[Tensor, "batch_size ... seq_len d_k"],
    V: Float[Tensor, "batch_size ... seq_len d_v"],
    mask: Bool[Tensor, "seq_len seq_len"] | None=None
) -> Float[Tensor, "batch_size ... seq_len d_v"]:
    d_k = Q.shape[-1]
    pre_softmax = einsum(Q, K, "... n d_k, ... m d_k -> ... n m") / math.sqrt(d_k)
    if mask is not None:
        pre_softmax = pre_softmax.masked_fill(~mask, float('-inf'))
    post_softmax = softmax(pre_softmax, -1)
    return einsum(post_softmax, V, "... n m, ... m d_v -> ... n d_v")

def cross_entropy(
    inputs: Float[Tensor, " batch_size vocab_size"],
    targets: Int[Tensor, " batch_size"]
) -> Float[Tensor, ""]:
    m = torch.max(inputs, dim=-1, keepdim=True)[0]
    exp = torch.exp(inputs - m)
    logsum = torch.log(torch.sum(exp, dim=-1, keepdim=True))
    return torch.mean(logsum - inputs.gather(-1, targets.unsqueeze(-1)) + m)

def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> Float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    elif it <= cosine_cycle_iters:
        return (min_learning_rate
            + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi))
                * (max_learning_rate - min_learning_rate))
    else:
        return min_learning_rate
