import argparse
import logging
import os
import einops
import torch
from cs336_basics.adamw import AdamW
from cs336_basics.functions import cross_entropy, data_loading, gradient_clipping, learning_rate_schedule, load_checkpoint, save_checkpoint
from cs336_basics.model.transformer_lm import TransformerLM
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    # TODO: either require=True, or specify default
    parser = argparse.ArgumentParser(description="Train Transformer LM")

    parser.add_argument("--path-train", default="data/tokens_tinystories_valid.npy")
    parser.add_argument("--path-state-dir", default="data/out")
    parser.add_argument("--path-state-src", required=False)

    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--rope-theta", type=float)
    parser.add_argument("--vocab-size", type=int)

    parser.add_argument("--beta1", type=float)
    parser.add_argument("--beta2", type=float)
    parser.add_argument("--cosine-cycle-iters", type=int)
    parser.add_argument("--eps", type=float)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--max-l2-norm", type=float)
    parser.add_argument("--max-lr", type=float)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--warmup-iters", type=int)
    parser.add_argument("--weight-decay", type=float)

    return parser.parse_args()


def main(args: argparse.Namespace):
    logger.info(f"All params:")
    for k, v in vars(args).items():
        logger.info("\t%s: %s", k, v)

    os.makedirs(args.path_state_dir, exist_ok=True)
    device = torch.device(args.device)

    logger.info(f"Data loading")
    dataset_train: npt.NDArray = np.load(args.path_train, mmap_mode='r')
    logger.info(f"Data loaded")

    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        device,
        dtype=torch.float32
    ).to(device)
    model.train()

    optimizer = AdamW(
        model.parameters(),
        0.0,
        args.weight_decay,
        (args.beta1, args.beta2),
        args.eps
    )

    start_it = 0
    if args.path_state_src:
        logger.info(f"Model loading")
        start_it = load_checkpoint(args.path_state_src, model, optimizer) + 1
        logger.info(f"Model loaded")

    for it in range(start_it, args.iterations):
        logger.info(f"[it={it}] Batch sampling")
        in_indices, tgt_indices = data_loading(
            dataset_train,
            args.batch_size,
            args.context_length,
            device
        )
        logger.info(f"[it={it}] Batch sampled")

        logger.info(f"[it={it}] Forwarding")
        out_logits = model(in_indices)
        logger.info(f"[it={it}] Forwarded")

        loss = cross_entropy(
            einops.rearrange(out_logits, "... seq vocab -> (... seq) vocab"),
            einops.rearrange(tgt_indices, "... seq -> (... seq)")
        )

        logger.info(f"[it={it}] loss = {loss.item()}")

        logger.info(f"[it={it}] Backwarding")
        optimizer.zero_grad()
        loss.backward()
        logger.info(f"[it={it}] Backwarded")

        gradient_clipping(model.parameters(), args.max_l2_norm)
        
        lr = learning_rate_schedule(
            it,
            args.max_lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_cycle_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        logger.info(f"[it={it}] Optimizing. lr = {lr}")
        optimizer.step()
        logger.info(f"[it={it}] Optimized")

        path_state = f"{args.path_state_dir}/{it}.pth"
        save_checkpoint(model, optimizer, it, path_state)
        logger.info(f"[it={it}] Saved to {path_state}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(parse_args())
