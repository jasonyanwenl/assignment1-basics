from jaxtyping import Float, Int
import numpy.typing as npt
import torch
from torch import Tensor

from cs336_basics.functions import softmax
from cs336_basics.model.transformer_lm import TransformerLM


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
    model.eval()
    with torch.no_grad():
        batch_outputs: Int[Tensor, "batch seq"] = torch.tensor(batch_inputs, dtype=torch.long).to(device)
        finished = torch.zeros(batch_inputs.shape[0], dtype=torch.bool, device=device)
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