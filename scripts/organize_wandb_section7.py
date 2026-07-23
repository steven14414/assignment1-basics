#!/usr/bin/env python3
"""Rename / group existing cs336-section7 wandb runs to the Section 7 naming scheme."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml

PROJECT = "cs336-section7"
LOW_TOKEN_BUDGET = 50_000_000
CTX = 256


def format_lr(lr) -> str:
    val = float(lr)
    exp = f"{val:.1e}"
    mantissa, exponent = exp.split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp_i = int(exponent)
    if exp_i == 0:
        return mantissa
    return f"{mantissa}e{exp_i}"


def load_config(run_dir: Path) -> dict | None:
    cfg_path = run_dir / "files" / "config.yaml"
    if not cfg_path.is_file():
        return None
    raw = yaml.safe_load(cfg_path.read_text())
    if not isinstance(raw, dict):
        return None
    return {k: v.get("value") if isinstance(v, dict) and "value" in v else v for k, v in raw.items()}


def infer_labels(cfg: dict) -> tuple[str, str] | None:
    ckpt = Path(str(cfg.get("checkpoint_dir", ""))).name
    if not ckpt or ckpt == ".":
        return None

    bs = int(cfg.get("batch_size", 0))
    lr = format_lr(cfg.get("lr", 0))
    max_iters = int(cfg.get("max_iters", 0))
    train_data = str(cfg.get("train_data", ""))
    tokens = bs * CTX * max_iters

    if ckpt.startswith("owt_") or "owt_train" in train_data:
        return "owt/full/main", f"bs{bs}_lr{lr}__{max_iters // 1000}k"

    tier = "low" if tokens <= LOW_TOKEN_BUDGET else "full"

    if ckpt.startswith("ts_ablate_"):
        variant = ckpt.removeprefix("ts_ablate_")
        variant = re.sub(r"_lr\w+$", "", variant)
        return f"ts/{tier}/ablation", f"{variant}__bs{bs}_lr{lr}"

    if "diverge" in ckpt:
        return f"ts/{tier}/lr_diverge", f"bs{bs}_lr{lr}"

    if re.fullmatch(r"ts_bs\d+_lr.+", ckpt):
        return f"ts/{tier}/batch_lr", f"bs{bs}_lr{lr}"

    if re.fullmatch(r"ts_bs\d+", ckpt):
        return f"ts/{tier}/batch", f"bs{bs}_lr{lr}"

    if ckpt.startswith("ts_lr"):
        return f"ts/{tier}/lr_sweep", f"bs{bs}_lr{lr}"

    if ckpt.startswith("ts_baseline"):
        return f"ts/{tier}/baseline", f"bs{bs}_lr{lr}"

    return None


def iter_local_runs(wandb_dir: Path):
    for cfg in sorted(wandb_dir.glob("run-*/files/config.yaml")):
        run_dir = cfg.parent.parent
        meta_path = cfg.parent / "wandb-metadata.json"
        if not meta_path.is_file():
            continue
        cfg_dict = load_config(run_dir)
        if not cfg_dict:
            continue
        if cfg_dict.get("wandb_project") != PROJECT:
            continue
        labels = infer_labels(cfg_dict)
        if not labels:
            continue
        meta = json.loads(meta_path.read_text())
        yield {
            "id": meta.get("run_id") or run_dir.name.split("-")[-1],
            "old_name": meta.get("name", ""),
            "group": labels[0],
            "name": labels[1],
            "ckpt": Path(str(cfg_dict.get("checkpoint_dir", ""))).name,
        }


def apply_remote(updates: list[dict], dry_run: bool) -> None:
    import wandb

    api = wandb.Api()
    entity = api.default_entity
    path_prefix = f"{entity}/{PROJECT}"

    for item in updates:
        run_path = f"{path_prefix}/{item['id']}"
        msg = f"{item['old_name']:40s} -> [{item['group']}] {item['name']}"
        if dry_run:
            print(f"[dry-run] {msg}")
            continue
        run = api.run(run_path)
        run.name = item["name"]
        run.group = item["group"]
        run.update()
        print(f"[updated] {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wandb-dir", type=Path, default=Path("wandb"))
    parser.add_argument("--apply", action="store_true", help="Push renames to wandb cloud")
    args = parser.parse_args()

    updates = list(iter_local_runs(args.wandb_dir))
    if not updates:
        print("No cs336-section7 runs found under wandb/")
        return

    print(f"Found {len(updates)} local runs in project {PROJECT}\n")
    if args.apply:
        apply_remote(updates, dry_run=False)
    else:
        for item in updates:
            print(
                f"{item['ckpt']:35s}  {item['old_name']:35s}  ->  "
                f"[{item['group']}] {item['name']}"
            )
        print(f"\nTotal: {len(updates)}. Re-run with --apply to update wandb cloud.")


if __name__ == "__main__":
    main()
