#!/usr/bin/env bash
# After wave-1 LR jobs finish, launch wave-2 (remaining LRs + diverge + batch + ablations).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export LOW_RESOURCE=1
GPU="${1:-0}"
# shellcheck source=scripts/section7_names.sh
source "$ROOT/scripts/section7_names.sh"

echo "[wave2] waiting for wave-1 jobs..."
while pgrep -f "train.py.*experiments/section7/ts_lr[13]em[45]" >/dev/null 2>&1; do
  sleep 120
done

echo "[wave2] starting remaining LR sweep (1e-3, 3e-3, 1e-2)"
for start in 4 5 6; do
  "$ROOT/run_section7.sh" lr-sweep "$start" "$GPU"
  while pgrep -f "train.py.*experiments/section7/ts_lr" >/dev/null 2>&1; do sleep 120; done
done

echo "[wave2] starting diverge runs sequentially"
for start in 0 1 2 3; do
  "$ROOT/run_section7.sh" lr-diverge "$start" "$GPU"
  while pgrep -f "train.py.*experiments/section7" >/dev/null 2>&1; do sleep 120; done
done

echo "[wave2] starting batch sweep (one gpu, sequential batch sizes via re-launch)"
for bs in 1 4 16; do
  iters=$((40960000 / (bs * 256)))
  warmup=$((iters / 10))
  ckpt="experiments/section7/ts_bs${bs}"
  mkdir -p "$ckpt"
  CUDA_VISIBLE_DEVICES="$GPU" nohup "$ROOT/.venv/bin/python" "$ROOT/train.py" \
    --train-data data/tokenized/tinystories_train.npy \
    --valid-data data/tokenized/tinystories_valid.npy \
    --checkpoint-dir "$ckpt" \
    --wandb-project cs336-section7 \
    --wandb-group "$(section7_wandb_group_ts batch)" \
    --wandb-run-name "$(section7_wandb_name_bs_lr "$bs" "3e-4")" \
    --batch-size "$bs" --max-iters "$iters" --lr 3e-4 \
    --warmup-iters "$warmup" --cosine-cycle-iters "$iters" \
    > "experiments/section7/logs/ts_bs${bs}.log" 2>&1
  while pgrep -f "train.py.*ts_bs${bs}" >/dev/null 2>&1; do sleep 120; done
done

echo "[wave2] starting ablations sequentially"
for spec in "no_rmsnorm --no-rmsnorm" "post_norm --post-norm" "no_rope --no-rope" "silu_ffn --ffn-type silu --d-ff 2048"; do
  set -- $spec
  name=$1; shift
  ckpt="experiments/section7/ts_ablate_${name}"
  iters=$((40960000 / (16 * 256)))
  warmup=$((iters / 10))
  mkdir -p "$ckpt"
  CUDA_VISIBLE_DEVICES="$GPU" nohup "$ROOT/.venv/bin/python" "$ROOT/train.py" \
    --train-data data/tokenized/tinystories_train.npy \
    --valid-data data/tokenized/tinystories_valid.npy \
    --checkpoint-dir "$ckpt" \
    --wandb-project cs336-section7 \
    --wandb-group "$(section7_wandb_group_ts ablation)" \
    --wandb-run-name "$(section7_wandb_name_ablate "$name" 16 "3e-4")" \
    --batch-size 16 --max-iters "$iters" --lr 3e-4 \
    --warmup-iters "$warmup" --cosine-cycle-iters "$iters" \
    "$@" \
    > "experiments/section7/logs/ts_ablate_${name}.log" 2>&1
  while pgrep -f "train.py.*ts_ablate_${name}" >/dev/null 2>&1; do sleep 120; done
done

echo "[wave2] generating text from best baseline checkpoint (lr=3e-4)"
"$ROOT/run_section7.sh" generate experiments/section7/ts_lr3em4/checkpoint_final.pt "$GPU" experiments/section7/generated_lr3em4.txt

echo "[wave2] all queued jobs finished."
