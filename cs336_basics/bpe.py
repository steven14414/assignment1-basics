import json
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, get_context
from pathlib import Path

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries
from tests.common import gpt2_bytes_to_unicode

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PARALLEL_MIN_BYTES = 1 << 20  # 1 MiB


class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = dict(vocab)
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes not in self.vocab.values():
                token_id = len(self.vocab)
                self.vocab[token_id] = token_bytes
                self.token_to_id[token_bytes] = token_id

        self._special_pattern = (
            re.compile("|".join(re.escape(token) for token in self.special_tokens)) if self.special_tokens else None
        )

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | Path,
        merges_filepath: str | Path,
        special_tokens: list[str] | None = None,
    ) -> "Tokenizer":
        byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

        def str_to_bytes(s: str) -> bytes:
            return bytes(byte_decoder[c] for c in s)

        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(token_id): str_to_bytes(token) for token, token_id in raw_vocab.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) == 2:
                    merges.append((str_to_bytes(parts[0]), str_to_bytes(parts[1])))
        return cls(vocab, merges, special_tokens)

    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self._encode_text(text)

    def _encode_text(self, text: str):
        if self._special_pattern is None:
            yield from self._encode_ordinary(text)
            return
        pos = 0
        for match in self._special_pattern.finditer(text):
            if match.start() > pos:
                yield from self._encode_ordinary(text[pos : match.start()])
            yield self.token_to_id[match.group().encode("utf-8")]
            pos = match.end()
        if pos < len(text):
            yield from self._encode_ordinary(text[pos:])

    def _encode_ordinary(self, text):
        for match in re.finditer(PAT, text):
            token = tuple(bytes([byte]) for byte in match.group().encode("utf-8"))
            for bpe_token in self._apply_merges(token):
                yield self.token_to_id[bpe_token]

    def _apply_merges(self, token: tuple[bytes, ...]) -> tuple[bytes, ...]:
        token_list = list(token)
        while len(token_list) >= 2:
            best_index = None
            best_rank = None
            for i, pair in enumerate(zip(token_list, token_list[1:])):
                rank = self.merge_ranks.get(pair, None)
                if rank is not None:
                    if best_rank is None or rank < best_rank:
                        best_index = i
                        best_rank = rank
            if best_rank is None:
                break
            else:
                token_list[best_index : best_index + 2] = [token_list[best_index] + token_list[best_index + 1]]
        return tuple(token_list)


def _pretokenize_text(text: str, special_tokens: list[str] | None) -> Counter[bytes]:
    if special_tokens is None:
        pieces = [text]
    else:
        sp = re.compile("|".join(re.escape(t) for t in special_tokens))
        pieces = sp.split(text)
    counts = Counter()
    for piece in pieces:
        for match in re.finditer(PAT, piece):
            counts[match.group().encode("utf-8")] += 1
    return counts


def _pretokenize_chunk(args: tuple[str, int, int, list[str] | None]) -> Counter[bytes]:
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")
    return _pretokenize_text(chunk, special_tokens)


def _count_pretokens_serial(input_path, special_tokens) -> Counter[bytes]:
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
    return _pretokenize_text(text, special_tokens)


def _count_pretokens(input_path, special_tokens) -> Counter[bytes]:
    input_path = str(input_path)
    file_size = os.path.getsize(input_path)
    if file_size < _PARALLEL_MIN_BYTES or not special_tokens:
        return _count_pretokens_serial(input_path, special_tokens)

    num_processes = min(cpu_count() or 1, 64)
    split_token = special_tokens[0].encode("utf-8")
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, split_token)

    if len(boundaries) <= 2:
        return _count_pretokens_serial(input_path, special_tokens)

    tasks = [
        (input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:], strict=False)
    ]
    counts: Counter[bytes] = Counter()
    with ProcessPoolExecutor(max_workers=len(tasks), mp_context=get_context("forkserver")) as executor:
        for partial in executor.map(_pretokenize_chunk, tasks, chunksize=1):
            counts.update(partial)
    return counts


def train_bpe(input_path, vocab_size: int, special_tokens: list[str]):
    vocab: dict[int, bytes] = {}
    for t in special_tokens:
        vocab[len(vocab)] = t.encode("utf-8")
    for b in range(256):
        vocab[len(vocab)] = bytes([b])
    num_merges = vocab_size - len(vocab)
    assert num_merges >= 0
    merges: list[tuple[bytes, bytes]] = []
    counts = _count_pretokens(input_path, special_tokens)
    word_counts: dict[tuple[bytes, ...], int] = {tuple(bytes([b]) for b in byte): cnt for byte, cnt in counts.items()}

    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for word, word_cnt in word_counts.items():
        for pair, cnt in Counter(zip(word, word[1:])).items():
            pair_counts[pair] += cnt * word_cnt
            pair_to_words[pair].add(word)

    for _ in range(num_merges):
        pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append((pair[0], pair[1]))
        vocab[len(vocab)] = pair[0] + pair[1]
        pair_counts.pop(pair)
        affected_words = pair_to_words.pop(pair)
        for word in affected_words:
            old_pairs = Counter(zip(word, word[1:]))
            word_cnt = word_counts.pop(word)
            new_word = _merge_word(word, pair)
            new_pairs = Counter(zip(new_word, new_word[1:]))
            word_counts[new_word] = word_cnt
            for p, cnt in old_pairs.items():
                if p == pair:
                    continue
                pair_counts[p] -= cnt * word_cnt
                pair_to_words[p].remove(word)
                if pair_counts[p] == 0:
                    pair_counts.pop(p)
                    pair_to_words.pop(p)
            for p, cnt in new_pairs.items():
                pair_counts[p] += cnt * word_cnt
                pair_to_words[p].add(new_word)

    return vocab, merges


def _merge_word(word, pair):
    new_word: list[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            new_word.append(word[i] + word[i + 1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)
