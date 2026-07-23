#!/usr/bin/env bash
# Section 7 experiment launcher. Runs training jobs on specified GPU(s).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

PYTHON="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing venv python at $PYTHON" >&2
  exit 1
fi

TOTAL_TOKENS=327680000
LOW_RESOURCE_TOKENS=40960000
CTX=256
TRAIN_DATA="data/tokenized/tinystories_train.npy"
VALID_DATA="data/tokenized/tinystories_valid.npy"
OWT_TRAIN="data/tokenized/owt_train.npy"
OWT_VALID="data/tokenized/owt_valid.npy"
EXP_DIR="experiments/section7"
mkdir -p "$EXP_DIR/logs"

# Shared GPUs (vLLM co-tenant): use smaller batches and fewer tokens.
if [[ "${LOW_RESOURCE:-0}" == "1" ]]; then
  DEFAULT_BS=16
  TOKEN_BUDGET="$LOW_RESOURCE_TOKENS"
else
  DEFAULT_BS=64
  TOKEN_BUDGET="$TOTAL_TOKENS"
fi

iters_for_batch() {
  local bs="$1"
  echo $((TOKEN_BUDGET / (bs * CTX)))
}

run_train() {
  local name="$1"
  local gpu="$2"
  shift 2
  local ckpt_dir="$EXP_DIR/$name"
  mkdir -p "$ckpt_dir"
  local log="$EXP_DIR/logs/${name}.log"

  echo "Starting $name on GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" train.py \
    --train-data "$TRAIN_DATA" \
    --valid-data "$VALID_DATA" \
    --checkpoint-dir "$ckpt_dir" \
    --wandb-project cs336-section7 \
    --wandb-run-name "$name" \
    "$@" \
    > "$log" 2>&1 &
  echo "$!" > "$EXP_DIR/logs/${name}.pid"
}

run_owt_train() {
  local name="$1"
  local gpu="$2"
  shift 2
  local ckpt_dir="$EXP_DIR/$name"
  mkdir -p "$ckpt_dir"
  local log="$EXP_DIR/logs/${name}.log"

  echo "Starting $name on GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PYTHON" train.py \
    --train-data "$OWT_TRAIN" \
    --valid-data "$OWT_VALID" \
    --vocab-size 32000 \
    --checkpoint-dir "$ckpt_dir" \
    --wandb-project cs336-section7 \
    --wandb-run-name "$name" \
    "$@" \
    > "$log" 2>&1 &
  echo "$!" > "$EXP_DIR/logs/${name}.pid"
}

case "${1:-help}" in
  baseline)
    GPU="${2:-0}"
    BS="${DEFAULT_BS}"
    ITERS=$(iters_for_batch "$BS")
    WARMUP=$((ITERS / 10))
    run_train "ts_baseline_bs${BS}_lr3e-4" "$GPU" \
      --batch-size "$BS" --max-iters "$ITERS" --lr 3e-4 \
      --warmup-iters "$WARMUP" --cosine-cycle-iters "$ITERS"
    ;;

  lr-sweep)
    START=0
    if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
      START="$2"
      shift
    fi
    shift
    LRS=(1e-5 3e-5 1e-4 3e-4 1e-3 3e-3 1e-2)
    BS="${DEFAULT_BS}"
    ITERS=$(iters_for_batch "$BS")
    WARMUP=$((ITERS / 10))
    i=0
    for gpu in "$@"; do
      idx=$((START + i))
      if [[ $idx -ge ${#LRS[@]} ]]; then break; fi
      lr="${LRS[$idx]}"
      lr_tag="${lr//./p}"
      lr_tag="${lr_tag/e-/em}"
      run_train "ts_lr${lr_tag}" "$gpu" \
        --batch-size "$BS" --max-iters "$ITERS" --lr "$lr" \
        --warmup-iters "$WARMUP" --cosine-cycle-iters "$ITERS"
      i=$((i + 1))
    done
    ;;

  lr-diverge)
    START=0
    if [[ "${2:-}" =~ ^[0-9]+$ ]]; then
      START="$2"
      shift
    fi
    shift
    LRS=(5e-3 1e-2 2e-2 5e-2)
    BS="${DEFAULT_BS}"
    ITERS=$(iters_for_batch "$BS")
    WARMUP=$((ITERS / 10))
    i=0
    for gpu in "$@"; do
      idx=$((START + i))
      if [[ $idx -ge ${#LRS[@]} ]]; then break; fi
      lr="${LRS[$idx]}"
      lr_tag="${lr//./p}"
      lr_tag="${lr_tag/e-/em}"
      run_train "ts_lr_diverge_${lr_tag}" "$gpu" \
        --batch-size "$BS" --max-iters "$ITERS" --lr "$lr" \
        --warmup-iters "$WARMUP" --cosine-cycle-iters "$ITERS"
      i=$((i + 1))
    done
    ;;

  batch-sweep)
    shift
    BSS=(1 4 16 64 128)
    if [[ "${1:-}" == --batches ]]; then
      shift
      IFS=',' read -ra BSS <<< "$1"
      shift
    fi
    i=0
    for gpu in "$@"; do
      if [[ $i -ge ${#BSS[@]} ]]; then break; fi
      bs="${BSS[$i]}"
      iters=$(iters_for_batch "$bs")
      warmup=$((iters / 10))
      run_train "ts_bs${bs}" "$gpu" \
        --batch-size "$bs" --max-iters "$iters" --lr 3e-4 \
        --warmup-iters "$warmup" --cosine-cycle-iters "$iters"
      i=$((i + 1))
    done
    ;;

  ablations)
    shift
    BS="${DEFAULT_BS}"
    ITERS=$(iters_for_batch "$BS")
    WARMUP=$((ITERS / 10))
    i=0
    for gpu in "$@"; do
      case "$i" in
        0)
          run_train "ts_ablate_no_rmsnorm" "$gpu" \
            --batch-size "$BS" --max-iters "$ITERS" --lr 3e-4 --no-rmsnorm \
            --warmup-iters "$WARMUP" --cosine-cycle-iters "$ITERS"
          ;;
        1)
          run_train "ts_ablate_post_norm" "$gpu" \
            --batch-size "$BS" --max-iters "$ITERS" --lr 3e-4 --post-norm \
            --warmup-iters "$WARMUP" --cosine-cycle-iters "$ITERS"
          ;;
        2)
          run_train "ts_ablate_no_rope" "$gpu" \
            --batch-size "$BS" --max-iters "$ITERS" --lr 3e-4 --no-rope \
            --warmup-iters "$WARMUP" --cosine-cycle-iters "$ITERS"
          ;;
        3)
          run_train "ts_ablate_silu_ffn" "$gpu" \
            --batch-size "$BS" --max-iters "$ITERS" --lr 3e-4 --ffn-type silu --d-ff 2048 \
            --warmup-iters "$WARMUP" --cosine-cycle-iters "$ITERS"
          ;;
        *) break ;;
      esac
      i=$((i + 1))
    done
    ;;

  owt)
    GPU="${2:-0}"
    BS="${DEFAULT_BS}"
    ITERS=$(iters_for_batch "$BS")
    WARMUP=$((ITERS / 10))
    run_owt_train "owt_baseline_bs${BS}" "$GPU" \
      --batch-size "$BS" --max-iters "$ITERS" --lr 3e-4 \
      --warmup-iters "$WARMUP" --cosine-cycle-iters "$ITERS"
    ;;

  generate)
    CKPT="${2:?checkpoint path required}"
    GPU="${3:-0}"
    OUT="${4:-$EXP_DIR/generated.txt}"
    CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON" generate_text.py \
      --checkpoint "$CKPT" \
      --output "$OUT" \
      --temperature 0.8 --top-p 0.95
    ;;

  status)
    echo "=== running section7 jobs ==="
    pgrep -af "train.py.*section7|train.py.*cs336-section7" || true
    echo
    echo "=== recent logs ==="
    ls -lt "$EXP_DIR/logs"/*.log 2>/dev/null | head -10 || true
    ;;

  *)
    cat <<'EOF'
Usage:
  ./run_section7.sh baseline [gpu]
  ./run_section7.sh lr-sweep gpu0 gpu1 ...
  ./run_section7.sh lr-diverge [start_idx] gpu0 gpu1 ...
  ./run_section7.sh batch-sweep gpu0 gpu1 ...
  ./run_section7.sh ablations gpu0 gpu1 gpu2 gpu3
  ./run_section7.sh owt [gpu]
  ./run_section7.sh generate <checkpoint> [gpu] [output]
  ./run_section7.sh status
EOF
    ;;
esac
