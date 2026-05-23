"""FLOP accounting for Transformer LM forward pass (matrix multiplies only)."""

from dataclasses import dataclass
from functools import reduce
import operator


VOCAB_SIZE = 50257
CONTEXT_LENGTH = 16384


@dataclass(frozen=True)
class ModelConfig:
    name: str
    num_layers: int
    d_model: int
    num_heads: int
    d_ff: int | None = None  # if None, round 8/3 * d_model to nearest multiple of 64

    @property
    def d_ff_resolved(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        target = 8 * self.d_model / 3
        return round(target / 64) * 64


GPT2_SMALL = ModelConfig("GPT-2 small", num_layers=12, d_model=768, num_heads=12)
GPT2_MEDIUM = ModelConfig("GPT-2 medium", num_layers=24, d_model=1024, num_heads=16)
GPT2_LARGE = ModelConfig("GPT-2 large", num_layers=36, d_model=1280, num_heads=20)
GPT2_XL = ModelConfig("GPT-2 XL", num_layers=48, d_model=1600, num_heads=25, d_ff=4288)


def flops(shape: tuple[int, ...]) -> int:
    """FLOPs for one GEMM: (m, n) @ (n, p) -> 2mnp."""
    return 2 * reduce(operator.mul, shape)


def flops_agg(shapes: list[tuple[int, ...]]) -> int:
    return sum(flops(s) for s in shapes)


def attn_flops(seq_len: int, d_model: int) -> dict[str, int]:
    """Per-layer attention GEMMs."""
    proj = flops_agg([(seq_len, d_model, d_model)] * 4)  # Q, K, V, O
    scores = flops_agg([
        (seq_len, d_model, seq_len),  # QK^T (all heads)
        (seq_len, seq_len, d_model),  # attn @ V (all heads)
    ])
    return {
        "attn_proj": proj,
        "attn_scores": scores,
        "attn": proj + scores,
    }


def ffn_flops(seq_len: int, d_model: int, d_ff: int) -> int:
    """Per-layer SwiGLU GEMMs (w1, w3, w2)."""
    return flops_agg([
        (seq_len, d_model, d_ff),
        (seq_len, d_model, d_ff),
        (seq_len, d_ff, d_model),
    ])


def lm_head_flops(seq_len: int, d_model: int, vocab_size: int = VOCAB_SIZE) -> int:
    return flops((seq_len, d_model, vocab_size))


def forward_flops(
    cfg: ModelConfig,
    seq_len: int = CONTEXT_LENGTH,
    vocab_size: int = VOCAB_SIZE,
) -> dict[str, int]:
    """Total forward-pass GEMM FLOPs for one sequence (batch size 1)."""
    per_layer_attn = attn_flops(seq_len, cfg.d_model)
    per_layer_ffn = ffn_flops(seq_len, cfg.d_model, cfg.d_ff_resolved)
    head = lm_head_flops(seq_len, cfg.d_model, vocab_size)

    attn_total = cfg.num_layers * per_layer_attn["attn"]
    ffn_total = cfg.num_layers * per_layer_ffn
    total = attn_total + ffn_total + head

    return {
        "attn_proj": cfg.num_layers * per_layer_attn["attn_proj"],
        "attn_scores": cfg.num_layers * per_layer_attn["attn_scores"],
        "attn": attn_total,
        "ffn": ffn_total,
        "lm_head": head,
        "total": total,
    }


def pct(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


def print_breakdown(cfg: ModelConfig, seq_len: int = CONTEXT_LENGTH) -> dict[str, int]:
    f = forward_flops(cfg, seq_len)
    total = f["total"]

    print(f"\n=== {cfg.name} ===")
    print(
        f"  layers={cfg.num_layers}, d_model={cfg.d_model}, heads={cfg.num_heads}, "
        f"d_ff={cfg.d_ff_resolved}, seq_len={seq_len}"
    )
    print(f"  Total GEMM FLOPs: {total:,}")

    rows = [
        ("attention (Q/K/V/O proj)", f["attn_proj"]),
        ("attention (QK^T + attn·V)", f["attn_scores"]),
        ("attention (total)", f["attn"]),
        ("FFN (SwiGLU)", f["ffn"]),
        ("LM head", f["lm_head"]),
    ]
    for label, flops_val in rows:
        print(f"  {label:28s} {flops_val:>18,}  ({pct(flops_val, total):5.1f}%)")

    return f


def print_comparison_table(configs: list[ModelConfig]) -> None:
    keys = ["attn", "ffn", "lm_head"]
    print("\n" + "=" * 72)
    print("FLOP share by component (% of forward-pass GEMMs, batch=1)")
    print("=" * 72)
    header = f"{'Model':<14}" + "".join(f"{k:>14}" for k in keys) + f"{'total':>16}"
    print(header)
    print("-" * len(header))

    for cfg in configs:
        f = forward_flops(cfg)
        total = f["total"]
        shares = [pct(f[k], total) for k in keys]
        row = f"{cfg.name:<14}" + "".join(f"{s:13.1f}%" for s in shares) + f"{total:>16,}"
        print(row)


if __name__ == "__main__":
    configs = [GPT2_SMALL, GPT2_MEDIUM, GPT2_LARGE, GPT2_XL]
    for cfg in configs:
        print_breakdown(cfg)
    print_comparison_table(configs)

    # Total trainable params: 1640452800 => if using fp32, ~6.1 GB  (params * 4 bytes)
        # if considering gradient: 6.1GB * 2 ~= 12.2GB
    # Total FLOPs: 3,516,769,894,400 (3.5 trillion)
        # attn: ~1.33T / 38%
        # ffn: ~2.02T / 58%
        # lm_head: ~0.165T / ~5%   # (your old note said 0.165B — that was a typo)
    # attn: 8Td^2 + 4T^2d
    # ffn: 4Tdd_ff + 2Tdd_ff ~= 16Td^2


