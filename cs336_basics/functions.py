from jaxtyping import Float
import torch


def softmax(x: Float[torch.Tensor, "..."], dim: int) -> Float[torch.Tensor, " ..."]:
    exp = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    return exp / torch.sum(exp, dim=dim, keepdim=True)