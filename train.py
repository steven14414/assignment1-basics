import argparse
import configparser
import os
import time
from pathlib import Path

import numpy as np
import torch
import wandb

from cs336_basics.adamw import AdamW
from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.data_loading import get_batch
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_schedule import lr_schedule
from cs336_basics.transformer_lm import TransformerLM

SETTINGS_PATH = Path(__file__).resolve().parent / "wandb" / "settings"


def load_wandb_settings(settings_path: Path = SETTINGS_PATH) -> dict[str, str]:
    if not settings_path.is_file():
        return {}

    parser = configparser.ConfigParser()
    parser.read(settings_path)
    if "default" not in parser:
        return {}

    section = parser["default"]
    return {key: section[key].strip() for key in ("api_key", "entity", "project") if section.get(key)}


def setup_wandb(args) -> None:
    settings = load_wandb_settings()
    if "WANDB_API_KEY" not in os.environ and (api_key := settings.get("api_key")):
        wandb.login(key=api_key, relogin=True)

    init_kwargs = {"project": args.wandb_project, "name": args.wandb_run_name, "config": vars(args)}
    if entity := settings.get("entity"):
        init_kwargs["entity"] = entity
    wandb.init(**init_kwargs)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    model.eval()
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        total_loss += cross_entropy(logits, y).item()
    model.train()
    return total_loss / num_batches


def log(step: int, train_loss: float, lr: float, elapsed: float, valid_loss: float | None = None) -> None:
    msg = f"step {step:6d} | train_loss {train_loss:.4f} | lr {lr:.2e} | elapsed {elapsed:.1f}s"
    if valid_loss is not None:
        msg += f" | valid_loss {valid_loss:.4f}"
    print(msg)

    if wandb.run is not None:
        metrics = {"train/loss": train_loss, "train/lr": lr, "time/elapsed_s": elapsed}
        if valid_loss is not None:
            metrics["valid/loss"] = valid_loss
        wandb.log(metrics, step=step)


def train(args):
    set_seed(args.seed)
    args.cosine_cycle_iters = args.cosine_cycle_iters or args.max_iters

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_data = np.load(args.train_data, mmap_mode="r")
    valid_data = np.load(args.valid_data, mmap_mode="r")

    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        args.rope_theta,
        use_rmsnorm=not args.no_rmsnorm,
        post_norm=args.post_norm,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
    ).to(args.device)
    optim = AdamW(model.parameters(), args.lr, args.weight_decay, (args.beta1, args.beta2), args.eps)

    start_it = 0
    if args.resume is not None:
        start_it = load_checkpoint(args.resume, model, optim) + 1
        print(f"resumed from {args.resume}, starting at step {start_it}")

    if not args.no_wandb:
        setup_wandb(args)

    start_time = time.perf_counter()

    for it in range(start_it, args.max_iters):
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        lr = lr_schedule(it, args.lr, args.min_lr, args.warmup_iters, args.cosine_cycle_iters)
        for group in optim.param_groups:
            group["lr"] = lr

        logits = model(x)
        loss = cross_entropy(logits, y)
        optim.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optim.step()

        valid_loss = None
        if it % args.eval_interval == 0:
            valid_loss = evaluate(
                model, valid_data, args.batch_size, args.context_length, args.device, args.eval_batches
            )

        if it % args.log_interval == 0 or valid_loss is not None:
            log(it, loss.item(), lr, time.perf_counter() - start_time, valid_loss)

        if it > 0 and it % args.checkpoint_interval == 0:
            ckpt_path = checkpoint_dir / f"checkpoint_{it}.pt"
            save_checkpoint(model, optim, it, ckpt_path)
            print(f"saved checkpoint to {ckpt_path}")

    final_it = args.max_iters - 1
    final_path = checkpoint_dir / "checkpoint_final.pt"
    save_checkpoint(model, optim, final_it, final_path)
    print(f"saved final checkpoint to {final_path}")

    if wandb.run is not None:
        wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(
        description="a script that runs a training loop to train your model on user-provided input."
    )
    # data
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--valid-data", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)

    # model
    parser.add_argument("--vocab-size", type=int, default=10000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # architecture ablations (section 7.3)
    parser.add_argument("--no-rmsnorm", action="store_true")
    parser.add_argument("--post-norm", action="store_true")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--ffn-type", type=str, default="swiglu", choices=["swiglu", "silu"])

    # training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-iters", type=int, default=20000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-batches", type=int, default=10)

    # optimizer + lr_schedule
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--warmup-iters", type=int, default=2000)
    parser.add_argument("--cosine-cycle-iters", type=int, default=None)  # None → 用 max_iters

    # logging
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--checkpoint-interval", type=int, default=5000)
    parser.add_argument("--wandb-project", type=str, default="cs336-basics")
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
