"""Profile BPE training and export visualizations.

Usage:
  uv run profile_bpe.py                          # debug dataset (fast)
  uv run profile_bpe.py --full                   # full TinyStories (~7 min + ~2x cProfile overhead)
  uv run profile_bpe.py --snakeviz               # also launch interactive snakeviz server
  uv run profile_bpe.py --port 8080 --snakeviz   # custom port for SSH port forwarding
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import subprocess
import sys
from pathlib import Path

from cs336_basics.bpe import train_bpe

PROFILES_DIR = Path("profiles")
DEBUG_INPUT = Path("tests/fixtures/tinystories_sample_5M.txt")
FULL_INPUT = Path("data/TinyStoriesV2-GPT4-train.txt")


def run_profile(input_path: Path, vocab_size: int, tag: str) -> Path:
    prof_path = PROFILES_DIR / f"bpe_{tag}.prof"
    PROFILES_DIR.mkdir(exist_ok=True)

    print(f"Profiling {input_path} (vocab_size={vocab_size}) -> {prof_path}")
    cProfile.runctx(
        "train_bpe(input_path, vocab_size, special_tokens=['<|endoftext|>'])",
        {"train_bpe": train_bpe, "input_path": input_path, "vocab_size": vocab_size},
        {},
        filename=str(prof_path),
    )
    return prof_path


def print_summary(prof_path: Path, limit: int = 15) -> None:
    stats = pstats.Stats(str(prof_path))
    print(f"\n=== Top {limit} by cumulative time ===")
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(limit)
    print(f"\n=== Functions in bpe.py ===")
    stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats("bpe.py")


def export_svg(prof_path: Path, tag: str) -> tuple[Path, Path]:
    flame_path = PROFILES_DIR / f"bpe_{tag}_flame.svg"
    callgraph_path = PROFILES_DIR / f"bpe_{tag}_callgraph.svg"

    subprocess.run(
        [sys.executable, "-m", "flameprof", str(prof_path)],
        check=True,
        stdout=flame_path.open("w"),
    )
    with callgraph_path.open("w") as out:
        gprof = subprocess.run(
            [sys.executable, "-m", "gprof2dot", "-f", "pstats", str(prof_path)],
            check=True,
            capture_output=True,
        )
        subprocess.run(["dot", "-Tsvg"], input=gprof.stdout, check=True, stdout=out)

    return flame_path, callgraph_path


def launch_snakeviz(prof_path: Path, port: int) -> None:
    print(f"\nStarting snakeviz on http://127.0.0.1:{port}")
    print("If you are on a remote machine, run locally:")
    print(f"  ssh -L {port}:localhost:{port} g27")
    print("Then open the URL in your browser.")
    subprocess.run([sys.executable, "-m", "snakeviz", "-s", "-H", "127.0.0.1", "-p", str(port), str(prof_path)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile BPE training and export visualizations.")
    parser.add_argument("--full", action="store_true", help="Use full TinyStories dataset.")
    parser.add_argument("--snakeviz", action="store_true", help="Launch snakeviz after profiling.")
    parser.add_argument("--port", type=int, default=8080, help="snakeviz port (default: 8080).")
    parser.add_argument("--skip-run", action="store_true", help="Only visualize an existing .prof file.")
    parser.add_argument("--prof", type=Path, help="Existing .prof file (with --skip-run).")
    args = parser.parse_args()

    if args.skip_run:
        if args.prof is None:
            parser.error("--skip-run requires --prof")
        prof_path = args.prof
        tag = prof_path.stem.removeprefix("bpe_")
    else:
        if args.full:
            input_path, vocab_size, tag = FULL_INPUT, 10000, "tinystories"
        else:
            input_path, vocab_size, tag = DEBUG_INPUT, 1000, "debug"
        prof_path = run_profile(input_path, vocab_size, tag)

    print_summary(prof_path)
    flame_path, callgraph_path = export_svg(prof_path, tag)
    print(f"\nSaved:")
    print(f"  profile:   {prof_path}")
    print(f"  flame:     {flame_path}")
    print(f"  callgraph: {callgraph_path}")

    if args.snakeviz:
        launch_snakeviz(prof_path, args.port)


if __name__ == "__main__":
    main()
