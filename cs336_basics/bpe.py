import regex as re
from collections import Counter, defaultdict

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _count_pretokens(input_path, special_tokens) -> Counter[bytes]:
    with open(input_path, encoding="utf-8") as f:
        text = f.read()
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
