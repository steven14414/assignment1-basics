import json
from pathlib import Path

from cs336_basics.bpe import train_bpe
from tests.common import gpt2_bytes_to_unicode

input_path = Path("data/TinyStoriesV2-GPT4-train.txt")
vocab_path = Path("data/tinystories_vocab.json")
merges_path = Path("data/tinystories_merges.txt")

byte_encoder = gpt2_bytes_to_unicode()


def bytes_to_str(b: bytes) -> str:
    return "".join(byte_encoder[x] for x in b)


vocab, merges = train_bpe(input_path, 10000, special_tokens=["<|endoftext|>"])

vocab_json = {bytes_to_str(token): tid for tid, token in vocab.items()}
with vocab_path.open("w", encoding="utf-8") as f:
    json.dump(vocab_json, f, indent=4, ensure_ascii=False)

with merges_path.open("w", encoding="utf-8") as f:
    for a, b in merges:
        f.write(f"{bytes_to_str(a)} {bytes_to_str(b)}\n")

longest_token = max(vocab.values(), key=len)
print(f"Saved vocab to {vocab_path}")
print(f"Saved merges to {merges_path}")
print(f"Longest token ({len(longest_token)} bytes): {longest_token.decode('utf-8', errors='replace')}")
