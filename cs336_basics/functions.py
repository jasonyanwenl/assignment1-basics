import math
import os
from typing import IO, BinaryIO, Iterable
from einops import einsum
from jaxtyping import Float, Bool, Int
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor

from cs336_basics.model.transformer_lm import TransformerLM


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

def gradient_clipping(params_itr: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    params = [p for p in params_itr if p.grad is not None]
    square_sum = sum(p.grad.pow(2).sum() for p in params)
    l2_norm = square_sum.sqrt()
    if l2_norm < max_l2_norm:
        return
    scale = max_l2_norm / (l2_norm + 1e-6)
    for param in params:
        param.grad.mul_(scale)

def data_loading(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size).reshape(-1, 1)
    offsets = np.arange(context_length)
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
    obj = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(obj, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    obj = torch.load(src, map_location=next(model.parameters()).device)
    model.load_state_dict(obj["model"])
    optimizer.load_state_dict(obj["optimizer"])
    return obj["iteration"]

def decoding(
    batch_inputs: npt.NDArray,
    model: TransformerLM,
    context_length: int,
    eot_token_id: int,
    max_tokens: int=100,
    temperature: float=1.0,
    top_p_thres: float=None
) -> npt.NDArray:
    device = next(model.parameters()).device
    with torch.no_grad():
        batch_outputs: Int[Tensor, "batch seq"] = torch.tensor(batch_inputs, dtype=torch.long).to(device)
        finished = torch.zeros(batch_inputs.shape[0], dtype=torch.bool, device=device)
        model.eval()
        for _ in range(min(max_tokens, max(0, context_length - batch_inputs.shape[1]))):
            if finished.all():
                break
            out_logits: Float[Tensor, "batch vocab"] = model(batch_outputs)[:, -1]
            prob = softmax(out_logits / temperature, dim=-1)

            if top_p_thres is not None:
                sorted_prob, token_ids_sorted_by_prob = prob.sort(dim=-1, descending=True)
                mask = torch.zeros_like(prob, dtype=torch.bool)
                mask[:, 1:] = sorted_prob.cumsum(dim=-1)[:, :-1] < top_p_thres
                mask[:, 0] = True
                raw_top_p = torch.where(mask, sorted_prob, 0.0)
                next_token_indices = torch.multinomial(raw_top_p / raw_top_p.sum(dim=-1, keepdim=True), num_samples=1)
                next_tokens: Int[Tensor, "batch 1"] = torch.gather(token_ids_sorted_by_prob, dim=-1, index=next_token_indices)
            else:
                next_tokens: Int[Tensor, "batch 1"] = torch.multinomial(prob, num_samples=1)

            finished |= (next_tokens == eot_token_id).squeeze(-1)
            batch_outputs = torch.cat((batch_outputs, next_tokens), dim=-1)
    return batch_outputs.numpy()
