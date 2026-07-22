"""Problem (tokenizer_experiments): part (d) — tokenize train/valid sets to uint16 .npy."""

from __future__ import annotations

import time
from array import array
from pathlib import Path

import numpy as np

from cs336_basics.bpe import Tokenizer

SPECIAL_TOKEN = "<|endoftext|>"
OUT_DIR = Path("data/tokenized")


def tokenize_file(tokenizer: Tokenizer, input_path: Path, output_path: Path) -> tuple[int, float]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ids: array = array("H")
    start = time.perf_counter()
    with input_path.open(encoding="utf-8") as f:
        for token_id in tokenizer.encode_iterable(f):
            ids.append(token_id)
    elapsed = time.perf_counter() - start
    arr = np.array(ids, dtype=np.uint16)
    np.save(output_path, arr)
    return len(arr), elapsed


def main() -> None:
    jobs = [
        ("data/TinyStoriesV2-GPT4-train.txt", "data/tinystories_vocab.json", "data/tinystories_merges.txt", "tinystories_train.npy"),
        ("data/TinyStoriesV2-GPT4-valid.txt", "data/tinystories_vocab.json", "data/tinystories_merges.txt", "tinystories_valid.npy"),
        ("data/owt_train.txt", "data/owt_vocab.json", "data/owt_merges.txt", "owt_train.npy"),
        ("data/owt_valid.txt", "data/owt_vocab.json", "data/owt_merges.txt", "owt_valid.npy"),
    ]

    for input_path, vocab_path, merges_path, out_name in jobs:
        tok = Tokenizer.from_files(vocab_path, merges_path, special_tokens=[SPECIAL_TOKEN])
        out_path = OUT_DIR / out_name
        num_tokens, elapsed = tokenize_file(tok, Path(input_path), out_path)
        file_bytes = Path(input_path).stat().st_size
        print(
            f"{out_name}: {num_tokens:,} tokens, "
            f"{file_bytes / elapsed:,.0f} bytes/s, {elapsed / 60:.1f} min -> {out_path}"
        )


if __name__ == "__main__":
    main()
