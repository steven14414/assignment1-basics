#!/usr/bin/env python3
"""Plot Section 7 full-scale LR sweep + edge-of-stability runs."""

from __future__ import annotations

from pathlib import Path

# Full-scale bs64, 20k steps — LR sweep (+ 6e-3 supplement)
SWEEP = [
    ("1e-5", 1e-5, 2.065),
    ("3e-5", 3e-5, 1.899),
    ("1e-4", 1e-4, 1.653),
    ("3e-4", 3e-4, 1.469),
    ("1e-3", 1e-3, 1.371),
    ("3e-3", 3e-3, 1.331),
    ("6e-3", 6e-3, 1.327),
    ("1e-2", 1e-2, 1.332),
]

# Edge of stability — same config, higher LR (7.2b)
DIVERGE = [
    ("2e-2", 2e-2, 1.358),
    ("5e-2", 5e-2, 1.608),
]

TARGET = 1.45
OUT_PNG = Path("figures/section7/lr_sweep_full.png")
OUT_SVG = Path("figures/section7/lr_sweep_full.svg")


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)

    sweep_lrs = [p[1] for p in SWEEP]
    sweep_vals = [p[2] for p in SWEEP]
    div_lrs = [p[1] for p in DIVERGE]
    div_vals = [p[2] for p in DIVERGE]

    # bridge sweep → diverge through shared 1e-2 point
    bridge_lrs = [sweep_lrs[-1]] + div_lrs
    bridge_vals = [sweep_vals[-1]] + div_vals

    all_lrs = sweep_lrs + div_lrs
    all_vals = sweep_vals + div_vals
    best_i = int(np.argmin(all_vals))

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.semilogx(sweep_lrs, sweep_vals, "o-", color="#2563eb", lw=2, ms=8, label="LR sweep")
    ax.semilogx(bridge_lrs, bridge_vals, "s--", color="#ea580c", lw=2, ms=9, label="edge of stability")

    ax.scatter([all_lrs[best_i]], [all_vals[best_i]], s=150, c="#dc2626", zorder=5,
               label=f"best: 6e-3 → {all_vals[best_i]:.3f}")

    for pts, color in ((SWEEP, "#2563eb"), (DIVERGE, "#ea580c")):
        for label, lr, val in pts:
            ax.annotate(f"{val:.3f}", (lr, val), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color=color)

    ax.axhline(TARGET, color="#9ca3af", ls="--", lw=1, label="target ≤ 1.45")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel("Final validation loss (per-token)")
    ax.set_title("TinyStories LR Sweep + Edge of Stability (full, batch=64, 327M tokens)")
    ax.set_xlim(8e-6, 6e-2)
    ax.set_ylim(1.2, 2.2)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    fig.savefig(OUT_SVG)
    print(f"saved {OUT_PNG}")
    print(f"saved {OUT_SVG}")


if __name__ == "__main__":
    main()
