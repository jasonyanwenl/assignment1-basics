
import math
from typing import Callable, Optional
import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay, betas, eps):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "weight_decay": weight_decay, "betas": betas, "eps": eps}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 1)
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                grad = p.grad.data
                adjusted_lr = lr * math.sqrt(1.0 - betas[1] ** t) / (1.0 - betas[0] ** t)
                p.data = (1.0 - lr * weight_decay) * p.data
                m = betas[0] * m + (1.0 - betas[0]) * grad
                v = betas[1] * v + (1.0 - betas[1]) * grad.pow(2)
                p.data -= adjusted_lr * m / (v.sqrt() + eps)
                state["t"] = t + 1
                state["m"] = m
                state["v"] = v
        return loss