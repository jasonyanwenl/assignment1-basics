import argparse
from datetime import datetime
import logging
import math
import os
import time
import einops
import torch
from tqdm import tqdm
import wandb
from cs336_basics import decoding
from cs336_basics.adamw import AdamW
from cs336_basics.functions import cross_entropy, data_loading, data_loading_by_starts, gradient_clipping, learning_rate_schedule, load_checkpoint, save_checkpoint
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
    parser.add_argument("--path-train", default="data/tokens_tinystories_train.npy")
    parser.add_argument("--path-vocab", default="vocab_tinystories.json")
    parser.add_argument("--path-merges", default="merges_tinystories.txt")

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
        device,
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
    checkpoint_every = max(args.iterations // 10, 10)
    model.to(device)

    logger.info("Loading eval batch indices")
    eval_batch_size = args.batch_size
    eval_last_idx = len(dataset_eval) - args.context_length  # exclusive
    total_eval_batches = (eval_last_idx + eval_batch_size - 1) // eval_batch_size
    eval_in_indices, eval_tgt_indices = data_loading_by_starts(
        dataset_eval,
        torch.linspace(0, eval_last_idx - 1, eval_batch_size).round().long().reshape(-1, 1),
        args.context_length,
        device
    )
    eval_every = max(args.iterations // 50, 10)
    logger.info("Loaded eval batch indices")

    logger.info("Loading tokenizer")
    peek_in_indices, _ = data_loading(dataset_eval, 1, int(args.context_length * 0.5), device)
    peek_in_indices = peek_in_indices.cpu().numpy()
    peek_every = max(args.iterations // 50, 10)
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.path_vocab,
        merges_filepath=args.path_merges,
        special_tokens=["<|endoftext|>"]
    )
    eot_id = tokenizer.vocab2id[b"<|endoftext|>"]
    logger.info("Loaded tokenizer")

    train_start = time.time()
    pbar = tqdm(range(start_it, args.iterations), desc="train", initial=start_it, total=args.iterations)
    for it in pbar:
        step_start = time.time()
        # logger.info(f"[it={it}] Batch sampling")
        in_indices, tgt_indices = data_loading(
            dataset_train,
            args.batch_size,
            args.context_length,
            device
        )
        # logger.info(f"[it={it}] Batch sampled")

        # logger.info(f"[it={it}] Forwarding")
        model.train()
        out_logits = model(in_indices)
        # logger.info(f"[it={it}] Forwarded")

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

        # logger.info(f"[it={it}] Backwarding")
        optimizer.zero_grad()
        loss.backward()
        # logger.info(f"[it={it}] Backwarded")

        l2_norm_before_grad_clip, grad_clip_scale = gradient_clipping(model.parameters(), args.max_l2_norm)

        # logger.info(f"[it={it}] Optimizing.")
        optimizer.step()
        # logger.info(f"[it={it}] Optimized")

        time_now = time.time()

        step_log = {
            "train/loss": loss.item(),
            "train/lr": lr,
            "train/l2_norm_before_grad_clip": l2_norm_before_grad_clip.item(),
            "train/grad_clip_scale": grad_clip_scale.item(),
            "time/elapsed_sec": time_now - train_start,
            "time/step_sec": time_now - step_start,
        }
        run.log(step_log, step=it)
        # logger.info(f"[it={it}] {step_log}")

        pbar.set_postfix(loss=f"{loss.item():.3f}", lr=f"{lr:.2e}")

        # Checkpoint
        if it == args.iterations - 1 or it % checkpoint_every == 0:
            path_state = f"{path_state_subfolder}/{it}.pth"
            save_checkpoint(model, optimizer, it, path_state)
            tqdm.write(f"[it={it}] Checkpointed to {path_state}")

        # Periodic eval
        if it in (start_it, args.iterations - 1) or it % eval_every == 0:
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
                tqdm.write(f"[it={it}] Eval: {eval_log}")
                run.log(eval_log, step=it)

        # Peek generation
        if it in (start_it, args.iterations - 1) or it % peek_every == 0:
            peek_out_indices = decoding.decoding(peek_in_indices, model, args.context_length, eot_id)
            for batch_idx in range(peek_in_indices.shape[0]):
                in_text = tokenizer.decode(peek_in_indices[batch_idx])
                out_text = tokenizer.decode(peek_out_indices[batch_idx])
                peek_log = {
                    "peek/in_text": in_text,
                    "peek/out_text": out_text
                }
                tqdm.write(f"[it={it}] [Peek decoding {batch_idx}] {peek_log}")
            run.log(peek_log, step=it)

    total_steps = args.iterations - start_it
    total_wall_sec = time.time() - train_start

    logger.info("Finished %d steps in %.1f sec (%.2f sec/step)",
                total_steps, total_wall_sec, total_wall_sec / total_steps)

    logger.info("Starting full eval")
    eval_loss, eval_perplexity = _eval_full(
        model, dataset_eval, total_eval_batches, eval_batch_size, eval_last_idx, args.context_length
    )
    logger.info(f"Finished full eval with eval_loss = {eval_loss}, eval_perplexity = {eval_perplexity}")

    run.summary["total_steps"] = total_steps
    run.summary["wall_clock_sec"] = total_wall_sec
    run.summary["sec_per_step"] = total_wall_sec / total_steps
    run.summary["eval_loss"] = eval_loss
    run.summary["eval_perplexity"] = eval_perplexity

    run.finish()


def _eval_full(
    model: TransformerLM, eval_dataset, total_eval_batches, eval_batch_size, eval_last_idx, context_length
) -> tuple[float, float]:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        eval_total_loss = 0.0
        eval_total_tokens = 0
        for eval_bidx in range(total_eval_batches):
            eval_start_idx = eval_bidx * eval_batch_size
            eval_end_idx = (
                eval_last_idx if eval_bidx == total_eval_batches - 1
                else (eval_bidx + 1) * eval_batch_size
            )
            eval_in_indices, eval_tgt_indices = data_loading_by_starts(
                eval_dataset,
                torch.arange(eval_start_idx, eval_end_idx).reshape(-1, 1),
                context_length,
                device
            )
            eval_out_logits = model(eval_in_indices)
            eval_step_loss = cross_entropy(
                einops.rearrange(eval_out_logits, "... seq vocab -> (... seq) vocab"),
                einops.rearrange(eval_tgt_indices, "... seq -> (... seq)")
            )
            eval_total_loss += eval_step_loss.item() * eval_tgt_indices.numel()
            eval_total_tokens += eval_tgt_indices.numel()
        eval_loss = eval_total_loss / eval_total_tokens
        eval_perplexity = math.exp(eval_loss)
    return (eval_loss, eval_perplexity)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(funcName)s:%(lineno)d %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    seed = 100000
    np.random.seed(seed)
    torch.manual_seed(seed)

    args = parse_args()

    logger.info(f"All params:")
    for k, v in vars(args).items():
        logger.info("\t%s: %s", k, v)

    main(args)
