#!/usr/bin/env bash
# Shared Section 7 checkpoint + wandb naming helpers.
# Checkpoint dirs keep legacy names (ts_lr3em3, …) for path compatibility.
# Wandb uses hierarchical groups + readable run names.

section7_tier_tag() {
  if [[ "${LOW_RESOURCE:-0}" == "1" ]]; then
    echo "low"
  else
    echo "full"
  fi
}

section7_lr_tag() {
  local lr="$1"
  local tag="${lr//./p}"
  tag="${tag/e-/em}"
  echo "$tag"
}

section7_wandb_group_ts() {
  local category="$1"
  echo "ts/$(section7_tier_tag)/${category}"
}

section7_wandb_group_owt() {
  echo "owt/full/main"
}

section7_wandb_name_bs_lr() {
  local bs="$1"
  local lr="$2"
  echo "bs${bs}_lr${lr}"
}

section7_wandb_name_ablate() {
  local variant="$1"
  local bs="$2"
  local lr="$3"
  echo "${variant}__bs${bs}_lr${lr}"
}

section7_wandb_name_owt() {
  local bs="$1"
  local lr="$2"
  local iters="$3"
  local k=$((iters / 1000))
  echo "bs${bs}_lr${lr}__${k}k"
}
