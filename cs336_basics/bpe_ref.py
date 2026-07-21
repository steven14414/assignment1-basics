from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from pathlib import Path

import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = dict(vocab)
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}
        self.merge_ranks = {pair: rank for rank, pair in enumerate(merges)}

        self.special_tokens = sorted(special_tokens or [], key=len, reverse=True)
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.token_to_id:
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
    ) -> Tokenizer:
        import json

        with open(vocab_filepath, encoding="utf-8") as f:
            raw_vocab = json.load(f)
        vocab = {int(token_id): token.encode("latin-1") for token_id, token in raw_vocab.items()}

        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split(" ")
                if len(parts) == 2:
                    merges.append((parts[0].encode("latin-1"), parts[1].encode("latin-1")))
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        return list(self.encode_iterable([text]))

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self._encode_text(text)

    def decode(self, ids: Iterable[int]) -> str:
        token_bytes = b"".join(self.vocab[token_id] for token_id in ids)
        return token_bytes.decode("utf-8", errors="replace")

    def _encode_text(self, text: str) -> Iterator[int]:
        if not text:
            return

        if self._special_pattern is None:
            yield from self._encode_ordinary(text)
            return

        start = 0
        for match in self._special_pattern.finditer(text):
            if match.start() > start:
                yield from self._encode_ordinary(text[start : match.start()])
            yield self.token_to_id[match.group(0).encode("utf-8")]
            start = match.end()
        if start < len(text):
            yield from self._encode_ordinary(text[start:])

    def _encode_ordinary(self, text: str) -> Iterator[int]:
        for match in re.finditer(PAT, text):
            token = tuple(bytes([byte]) for byte in match.group(0).encode("utf-8"))
            for bpe_token in self._apply_merges(token):
                yield self.token_to_id[bpe_token]

    def _apply_merges(self, token: tuple[bytes, ...]) -> tuple[bytes, ...]:
        if len(token) < 2:
            return token

        token_list = list(token)
        while len(token_list) >= 2:
            best_rank = None
            best_index = None
            for i, pair in enumerate(zip(token_list, token_list[1:], strict=False)):
                rank = self.merge_ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_index = i
            if best_index is None:
                break
            token_list[best_index : best_index + 2] = [token_list[best_index] + token_list[best_index + 1]]
        return tuple(token_list)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    del kwargs

    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    vocab: dict[int, bytes] = {i: token for i, token in enumerate(special_token_bytes)}
    for byte in range(256):
        vocab[len(vocab)] = bytes([byte])

    num_merges = vocab_size - len(vocab)
    if num_merges <= 0:
        return dict(list(vocab.items())[:vocab_size]), []

    pretokens = _count_pretokens(input_path, special_tokens)
    word_counts: dict[tuple[bytes, ...], int] = {
        tuple(bytes([byte]) for byte in token): count for token, count in pretokens.items()
    }

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = defaultdict(set)
    for word, count in word_counts.items():
        for pair in set(zip(word, word[1:], strict=False)):
            pair_counts[pair] += _count_pair_in_word(word, pair) * count
            pair_to_words[pair].add(word)

    merges: list[tuple[bytes, bytes]] = []
    for _ in range(num_merges):
        if not pair_counts:
            break

        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        if pair_counts[best_pair] <= 0:
            break

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        affected_words = list(pair_to_words.pop(best_pair, set()))
        pair_counts.pop(best_pair, None)

        for old_word in affected_words:
            count = word_counts.pop(old_word, 0)
            if count == 0:
                continue

            old_pairs = Counter(zip(old_word, old_word[1:], strict=False))
            new_word = _merge_word(old_word, best_pair)
            new_pairs = Counter(zip(new_word, new_word[1:], strict=False))
            word_counts[new_word] = word_counts.get(new_word, 0) + count

            for pair, occurrences in old_pairs.items():
                if pair == best_pair:
                    continue
                pair_counts[pair] -= occurrences * count
                if pair_counts[pair] <= 0:
                    pair_counts.pop(pair, None)
                word_set = pair_to_words.get(pair)
                if word_set is not None:
                    word_set.discard(old_word)
                    if not word_set:
                        pair_to_words.pop(pair, None)

            for pair, occurrences in new_pairs.items():
                pair_counts[pair] += occurrences * count
                pair_to_words[pair].add(new_word)

    return vocab, merges


def _count_pretokens(input_path: str | Path, special_tokens: list[str]) -> Counter[bytes]:
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        special_pattern = re.compile("|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)))
        pieces = special_pattern.split(text)
    else:
        pieces = [text]

    counts: Counter[bytes] = Counter()
    for piece in pieces:
        for match in re.finditer(PAT, piece):
            counts[match.group(0).encode("utf-8")] += 1
    return counts


def _count_pair_in_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> int:
    return sum(1 for candidate in zip(word, word[1:], strict=False) if candidate == pair)


def _merge_word(word: tuple[bytes, ...], pair: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    merged: list[bytes] = []
    i = 0
    while i < len(word):
        if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
            merged.append(word[i] + word[i + 1])
            i += 2
        else:
            merged.append(word[i])
            i += 1
    return tuple(merged)
