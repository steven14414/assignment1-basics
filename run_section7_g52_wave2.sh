#!/usr/bin/env bash
# g52 full-config wave-2: remaining 3 LRs in parallel + baseline
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "[g52-wave2] waiting for wave-1 (lr 1e-5..3e-4)..."
while pgrep -f "train.py.*experiments/section7/ts_lr[13]em[45]" >/dev/null 2>&1; do
  sleep 120
done

echo "[g52-wave2] remaining LR sweep (1e-3, 3e-3, 1e-2) on GPU 4 5 6"
"$ROOT/run_section7.sh" lr-sweep 4 4 5 6

while pgrep -f "train.py.*experiments/section7/ts_lr" >/dev/null 2>&1; do
  sleep 120
done

echo "[g52-wave2] full baseline on GPU 4"
"$ROOT/run_section7.sh" baseline 4

echo "[g52-wave2] done."
