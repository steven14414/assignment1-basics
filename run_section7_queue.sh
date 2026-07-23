#!/usr/bin/env bash
# Queue remaining Section 7 jobs sequentially on one GPU (for shared-gpu hosts).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export LOW_RESOURCE=1
GPU="${1:-0}"

wait_for_slot() {
  while pgrep -f "train.py.*experiments/section7" >/dev/null; do
    sleep 60
  done
}

run_and_wait() {
  "$ROOT/run_section7.sh" "$1" "$GPU"
  wait_for_slot
}

# Wave 1: LR sweep (all 7, one at a time)
for start in 0 1 2 3 4 5 6; do
  run_and_wait lr-sweep "$start" "$GPU"
done

# Wave 2: divergent LRs
for start in 0 1 2 3; do
  run_and_wait lr-diverge "$start" "$GPU"
done

# Wave 3: batch sizes (use smaller batches on shared GPU)
"$ROOT/run_section7.sh" batch-sweep "$GPU"
wait_for_slot

# Wave 4: ablations one by one
for i in 0 1 2 3; do
  "$ROOT/run_section7.sh" ablations "$GPU"
  wait_for_slot
done

echo "Section 7 queue finished."
