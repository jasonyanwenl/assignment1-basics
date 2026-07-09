import logging
import math
import os
from typing import IO, BinaryIO, Iterable
from einops import einsum
from jaxtyping import Float, Bool, Int
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


logger = logging.getLogger(__name__)

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

def gradient_clipping(params_itr: Iterable[torch.nn.Parameter], max_l2_norm: float) -> tuple[torch.Tensor, torch.Tensor]:
    params = [p for p in params_itr if p.grad is not None]
    if not params:
        return (torch.tensor(0.0), torch.tensor(1.0))
    square_sum = sum(p.grad.pow(2).sum() for p in params)
    l2_norm = square_sum.sqrt()
    scale = l2_norm.new_tensor(1.0)
    if l2_norm >= max_l2_norm:
        scale = max_l2_norm / (l2_norm + 1e-6)
        for param in params:
            param.grad.mul_(scale)
    return (l2_norm, scale)

def data_loading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
    starts = torch.randint(0, len(dataset) - context_length, size=(batch_size, 1))
    return data_loading_by_starts(dataset, starts, context_length, device)

def data_loading_by_starts(
    dataset: npt.NDArray, starts: torch.Tensor, context_length: int, device: str
) -> tuple[Int[Tensor, "batch seq"], Int[Tensor, "batch seq"]]:
    offsets = torch.arange(context_length)
    return (
        torch.tensor(dataset[starts + offsets], dtype=torch.long).to(device),
        torch.tensor(dataset[starts + offsets + 1], dtype=torch.long).to(device)
    )

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
):
    logger.info(f"Saving checkpoint to: {out}")
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)
    logger.info(f"Saved checkpoint to: {out}")

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    logger.info(f"Loading checkpoint from: {src}")
    obj = torch.load(src, map_location=next(model.parameters()).device)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    logger.info(f"Loaded checkpoint from: {src}")
    return obj["iteration"]
