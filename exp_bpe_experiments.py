"""Problem (tokenizer_experiments): parts (a)(b)(c)."""

from __future__ import annotations

import time
from array import array
from pathlib import Path

from cs336_basics.bpe import Tokenizer

SPECIAL_TOKEN = "<|endoftext|>"
NUM_DOCS = 10

PATHS = {
    "ts_train": Path("data/TinyStoriesV2-GPT4-train.txt"),
    "ts_valid": Path("data/TinyStoriesV2-GPT4-valid.txt"),
    "owt_train": Path("data/owt_train.txt"),
    "owt_valid": Path("data/owt_valid.txt"),
    "ts_vocab": Path("data/tinystories_vocab.json"),
    "ts_merges": Path("data/tinystories_merges.txt"),
    "owt_vocab": Path("data/owt_vocab.json"),
    "owt_merges": Path("data/owt_merges.txt"),
}


def sample_documents(path: Path, n: int = NUM_DOCS) -> list[str]:
    text = path.read_text(encoding="utf-8")
    docs = [doc for doc in text.split(SPECIAL_TOKEN) if doc.strip()]
    return docs[:n]


def compression_ratio(tokenizer: Tokenizer, documents: list[str]) -> float:
    num_bytes = 0
    num_tokens = 0
    for doc in documents:
        doc_bytes = doc.encode("utf-8")
        num_bytes += len(doc_bytes)
        num_tokens += len(tokenizer.encode(doc))
    return num_bytes / num_tokens


def measure_throughput(tokenizer: Tokenizer, path: Path) -> tuple[float, float]:
    with path.open(encoding="utf-8") as f:
        text = f.read()
    data = text.encode("utf-8")
    start = time.perf_counter()
    token_ids = tokenizer.encode(text)
    elapsed = time.perf_counter() - start
    return len(data) / elapsed, len(data) / len(token_ids)


def main() -> None:
    ts_tok = Tokenizer.from_files(PATHS["ts_vocab"], PATHS["ts_merges"], special_tokens=[SPECIAL_TOKEN])
    owt_tok = Tokenizer.from_files(PATHS["owt_vocab"], PATHS["owt_merges"], special_tokens=[SPECIAL_TOKEN])

    ts_docs = sample_documents(PATHS["ts_valid"])
    owt_docs = sample_documents(PATHS["owt_valid"])

    ts_on_ts = compression_ratio(ts_tok, ts_docs)
    owt_on_owt = compression_ratio(owt_tok, owt_docs)
    ts_on_owt = compression_ratio(ts_tok, owt_docs)
    owt_on_ts = compression_ratio(owt_tok, ts_docs)

    throughput, _ = measure_throughput(ts_tok, PATHS["ts_valid"])
    pile_seconds = 825e9 / throughput
    pile_days = pile_seconds / 86400

    print("=== (a) compression ratio on 10-document samples ===")
    print(f"TinyStories tokenizer on TinyStories: {ts_on_ts:.4f} bytes/token")
    print(f"OpenWebText tokenizer on OpenWebText: {owt_on_owt:.4f} bytes/token")

    print("\n=== (b) cross-domain compression ratio ===")
    print(f"TinyStories tokenizer on OpenWebText: {ts_on_owt:.4f} bytes/token")
    print(f"OpenWebText tokenizer on TinyStories: {owt_on_ts:.4f} bytes/token")

    print("\n=== (c) throughput (TinyStories valid, encode) ===")
    print(f"throughput: {throughput:,.0f} bytes/s")
    print(f"Pile 825GB estimate: {pile_days:.1f} days ({pile_seconds / 3600:.1f} hours)")


if __name__ == "__main__":
    main()
