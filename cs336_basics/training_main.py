import argparse
from datetime import datetime
import logging
import os
import time
import einops
import torch
import wandb
from cs336_basics import decoding
from cs336_basics.adamw import AdamW
from cs336_basics.functions import cross_entropy, data_loading, gradient_clipping, learning_rate_schedule, load_checkpoint, save_checkpoint
from cs336_basics.model.transformer_lm import TransformerLM
import numpy as np
import numpy.typing as npt

from cs336_basics.tokenizer.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer LM")

    parser.add_argument("--path-eval", default="data/tokens_tinystories_valid.npy")
    parser.add_argument("--path-state-dir", default="data/out")
    parser.add_argument("--path-state-src", required=False)
    parser.add_argument("--path-train", default="data/tokens_tinystories_valid.npy")

    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--device", type=torch.device, default=torch.device("cpu"))
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--vocab-size", type=int, default=10000)

    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--cosine-cycle-iters", type=int, default=6)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--max-l2-norm", type=float, default=1e-2)
    parser.add_argument("--max-lr", type=float, default=1e-2)
    parser.add_argument("--min-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    return parser.parse_args()


def main(args: argparse.Namespace):
    run = wandb.init(
        entity="lyw1124278064-personal",
        project="cs336-assignment-1",
        config=vars(args),
    )

    os.makedirs(args.path_state_dir, exist_ok=True)
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path_state_subfolder = f"{args.path_state_dir}/{run_ts}"
    os.makedirs(path_state_subfolder, exist_ok=True)

    device = torch.device(args.device)

    logger.info(f"Data loading")
    dataset_train: npt.NDArray = np.load(args.path_train, mmap_mode='r')
    dataset_eval: npt.NDArray = np.load(args.path_eval, mmap_mode='r')
    logger.info(f"Data loaded")

    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        args.device,
        dtype=torch.float32
    )

    optimizer = AdamW(
        model.parameters(),
        0.0,
        args.weight_decay,
        (args.beta1, args.beta2),
        args.eps
    )

    start_it = 0
    if args.path_state_src:
        start_it = load_checkpoint(args.path_state_src, model, optimizer) + 1

    model.to(device)

    eval_in_indices, eval_tgt_indices = data_loading(
        dataset_eval,
        args.batch_size,
        args.context_length,
        device
    )

    # TODO: Monitor the norms of activations, model weights, and gradients to make sure they are not exploding or vanishing

    train_start = time.time()

    for it in range(start_it, args.iterations):
        step_start = time.time()
        logger.info(f"[it={it}] Batch sampling")
        in_indices, tgt_indices = data_loading(
            dataset_train,
            args.batch_size,
            args.context_length,
            device
        )
        logger.info(f"[it={it}] Batch sampled")

        logger.info(f"[it={it}] Forwarding")
        model.train()
        out_logits = model(in_indices)
        logger.info(f"[it={it}] Forwarded")

        loss = cross_entropy(
            einops.rearrange(out_logits, "... seq vocab -> (... seq) vocab"),
            einops.rearrange(tgt_indices, "... seq -> (... seq)")
        )

        lr = learning_rate_schedule(
            it,
            args.max_lr,
            args.min_lr,
            args.warmup_iters,
            args.cosine_cycle_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        logger.info(f"[it={it}] Backwarding")
        optimizer.zero_grad()
        loss.backward()
        logger.info(f"[it={it}] Backwarded")

        gradient_clipping(model.parameters(), args.max_l2_norm)

        logger.info(f"[it={it}] Optimizing.")
        optimizer.step()
        logger.info(f"[it={it}] Optimized")

        time_now = time.time()

        step_log = {
            "train/loss": loss.item(),
            "train/lr": lr,
            "time/elapsed_sec": time_now - train_start,
            "time/step_sec": time_now - step_start,
        }
        run.log(step_log, step=it)
        logger.info(f"[it={it}] {step_log}")

        if it == args.iterations - 1 or it % 500 == 0:
            path_state = f"{path_state_subfolder}/{it}.pth"
            save_checkpoint(model, optimizer, it, path_state)
            logger.info(f"[it={it}] Saved to {path_state}")

        if it == args.iterations - 1 or it % 100 == 0:
            model.eval()
            with torch.no_grad():
                eval_out_logits = model(eval_in_indices)
                eval_loss = cross_entropy(
                    einops.rearrange(eval_out_logits, "... seq vocab -> (... seq) vocab"),
                    einops.rearrange(eval_tgt_indices, "... seq -> (... seq)")
                )
                eval_perplexity = eval_loss.exp()
                eval_log = {
                    "eval/loss": eval_loss.item(),
                    "eval/perplexity": eval_perplexity.item(),
                }
                logger.info(f"[it={it}] Eval: {eval_log}")
                run.log(eval_log, step=it)

    total_steps = args.iterations - start_it
    total_wall_sec = time.time() - train_start

    logger.info("Finished %d steps in %.1f sec (%.2f sec/step)",
                total_steps, total_wall_sec, total_wall_sec / total_steps)

    run.summary["total_steps"] = total_steps
    run.summary["wall_clock_sec"] = total_wall_sec
    run.summary["sec_per_step"] = total_wall_sec / total_steps

    run.finish()


def peek_decoding(args: argparse.Namespace):
    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        args.device,
        dtype=torch.float32
    )

    optimizer = AdamW(
        model.parameters(),
        0.0,
        args.weight_decay,
        (args.beta1, args.beta2),
        args.eps
    )

    path_state_src = args.path_state_src if args.path_state_src else "data/out/20260707_133459/199.pth"
    load_checkpoint(path_state_src, model, optimizer)

    vocab_filepath="vocab_tinystories.json"
    merges_filepath="merges_tinystories.txt"
    special_tokens=["<|endoftext|>"]
    tokenizer = Tokenizer.from_files(
        vocab_filepath=vocab_filepath,
        merges_filepath=merges_filepath,
        special_tokens=special_tokens
    )

    dataset_train: npt.NDArray = np.load("data/tokens_tinystories_valid.npy", mmap_mode='r')
    in_indices, tgt_indices = data_loading(dataset_train, 1, 100, "cpu")
    in_indices, tgt_indices = in_indices.numpy(), tgt_indices.numpy()
    out_indices = decoding.decoding(in_indices, model, args.context_length,
        tokenizer.vocab2id[b"<|endoftext|>"])

    logger.info("Peek decoding:\n")
    for batch_idx in range(in_indices.shape[0]):
        in_text = tokenizer.decode(in_indices[batch_idx])
        tgt_text = tokenizer.decode(tgt_indices[batch_idx])
        out_text = tokenizer.decode(out_indices[batch_idx])
        logger.info(f"[{batch_idx}] in_text:\n{in_text}\nout_text:\n{out_text}\ntgt_text:\n{tgt_text}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    seed = 100000
    np.random.seed(seed)
    torch.manual_seed(seed)

    args = parse_args()

    logger.info(f"All params:")
    for k, v in vars(args).items():
        logger.info("\t%s: %s", k, v)

    main(args)
    # peek_decoding(args)
