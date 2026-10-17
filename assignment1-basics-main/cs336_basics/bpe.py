import re
from collections import Counter
import regex

from cs336_basics.pretokenization_example import find_chunk_boundaries

def bpe(
    input_path: str,
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if any(token == "" for token in special_tokens):
        raise ValueError("Special tokens must not be empty")

    # Escape regex metacharacters and match longer tokens first when they overlap.
    special_token_pattern = "|".join(
        re.escape(token) for token in sorted(set(special_tokens), key=len, reverse=True)
    )
    total_counts = Counter()
    
    with open(input_path, "rb") as f:
        if special_tokens:
            chunk_boundaries = find_chunk_boundaries(f, 8, special_tokens[0].encode("utf-8"))
        else:
            f.seek(0, 2)
            chunk_boundaries = [0, f.tell()]
        f.seek(chunk_boundaries[0])
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:]):
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            # Keep document fragments separate so merges cannot cross special tokens.
            chunks = re.split(special_token_pattern, chunk) if special_token_pattern else [chunk]
            chunks = [part for part in chunks if part]

            # pretokenization this chunk
            counts = pretokenization(chunks)
            total_counts.update(counts)
            
    return train_bpe(total_counts, vocab_size, special_tokens)


def pretokenization(chunk: list[str]) -> dict[str, int]:
    pattern = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    counts = Counter()
    for part in chunk:
        for match in regex.finditer(pattern, part):
            token = match.group()
            counts[token] += 1

    return dict(counts)

def train_bpe(
    counts: dict[str, int],
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    pair_counts = Counter()
    word_tokens = {}

    for word, count in counts.items():
        word_bytes = word.encode("utf-8")
        tokens = [bytes([byte]) for byte in word_bytes]

        for left, right in zip(tokens[:-1], tokens[1:]):
            pair_counts[(left, right)] += count

        # Keep the token sequence so it can be updated after each merge.
        word_tokens[tuple(tokens)] = count

    vocab = {i: bytes([i]) for i in range(256)}
    for token in dict.fromkeys(special_tokens):
        if not token:
            raise ValueError("Special tokens must not be empty")
        vocab[len(vocab)] = token.encode("utf-8")

    if vocab_size < len(vocab):
        raise ValueError("vocab_size must fit all 256 bytes and special tokens")

    merges = []

    while len(vocab) < vocab_size:
        if not pair_counts:
            break

        # Break frequency ties by choosing the lexicographically largest pair.
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        merged_token = best_pair[0] + best_pair[1]
        merges.append(best_pair)
        vocab[len(vocab)] = merged_token

        updated_words = Counter()
        for tokens, count in word_tokens.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            updated_words[tuple(new_tokens)] += count

        word_tokens = updated_words

        # Recount pairs using the merged token sequences, rather than raw bytes.
        pair_counts = Counter()
        for tokens, count in word_tokens.items():
            for left, right in zip(tokens[:-1], tokens[1:]):
                pair_counts[(left, right)] += count

    return vocab, merges
