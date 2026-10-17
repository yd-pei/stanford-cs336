"""BPE with cppyy: Python handles regex, C++ handles counting and merges."""

import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import regex

from cs336_basics.pretokenization_example import find_chunk_boundaries
import cppyy

cppyy.cppdef(r"""
#include <map>
#include <string>
#include <vector>
#include <stdexcept>

namespace bpe_core {
using Counts = std::map<std::string, long long>;
using Pair = std::pair<std::string, std::string>;

void count_tokens(const std::vector<std::string>& tokens, Counts& counts) {
    for (const auto& token : tokens) counts[token]++;
}

struct Result {
    std::vector<std::string> vocab;
    std::vector<Pair> merges;
};

Result train(const Counts& counts, int vocab_size,
             const std::vector<std::string>& special_tokens) {
    Result result;
    for (int i = 0; i < 256; i++)
        result.vocab.push_back(std::string(1, static_cast<char>(i)));
    std::map<std::string, bool> seen;
    for (const auto& token : special_tokens) {
        if (token.empty()) throw std::invalid_argument("Special tokens must not be empty");
        if (!seen[token]) {
            result.vocab.push_back(token);
            seen[token] = true;
        }
    }
    if (vocab_size < static_cast<int>(result.vocab.size()))
        throw std::invalid_argument("vocab_size must fit all 256 bytes and special tokens");

    std::vector<std::vector<std::string>> words;
    std::vector<long long> frequencies;
    for (const auto& [word, count] : counts) {
        std::vector<std::string> tokens;
        for (char byte : word) tokens.push_back(std::string(1, byte));
        words.push_back(tokens);
        frequencies.push_back(count);
    }

    while (static_cast<int>(result.vocab.size()) < vocab_size) {
        std::map<Pair, long long> pair_counts;
        for (size_t w = 0; w < words.size(); w++)
            for (size_t i = 0; i + 1 < words[w].size(); i++)
                pair_counts[{words[w][i], words[w][i + 1]}] += frequencies[w];
        if (pair_counts.empty()) break;

        auto best = pair_counts.begin();
        for (auto it = pair_counts.begin(); it != pair_counts.end(); ++it)
            if (it->second > best->second ||
                (it->second == best->second && it->first > best->first))
                best = it;
        const auto& pair = best->first;
        std::string merged = pair.first + pair.second;
        result.vocab.push_back(merged);
        result.merges.push_back(pair);

        for (auto& tokens : words) {
            size_t out = 0;
            for (size_t i = 0; i < tokens.size(); i++) {
                if (i + 1 < tokens.size() && tokens[i] == pair.first && tokens[i + 1] == pair.second) {
                    tokens[out++] = merged;
                    i++;
                } else {
                    tokens[out++] = tokens[i];
                }
            }
            tokens.resize(out);
        }
    }
    return result;
}
}
""")

_core = cppyy.gbl.bpe_core


def bpe_cpp(
    input_path: str,
    vocab_size: int, 
    special_tokens: list[str],
    num_processes: int = 8,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if num_processes < 1:
        raise ValueError("num_processes must be positive")
    if any(token == "" for token in special_tokens):
        raise ValueError("Special tokens must not be empty")

    # Escape regex metacharacters and match longer tokens first when they overlap.
    special_token_pattern = "|".join(
        re.escape(token) for token in sorted(set(special_tokens), key=len, reverse=True)
    )
    total_counts = Counter()
    
    with open(input_path, "rb") as f:
        if special_tokens:
            chunk_boundaries = find_chunk_boundaries(f, num_processes, special_tokens[0].encode("utf-8"))
        else:
            f.seek(0, 2)
            chunk_boundaries = [0, f.tell()]
    jobs = [
        (str(input_path), start, end, special_token_pattern)
        for start, end in zip(chunk_boundaries[:-1], chunk_boundaries[1:])
    ]
    if len(jobs) <= 1:
        for job in jobs:
            total_counts.update(_pretokenize_chunk(job))
    else:
        # Send offsets, not file contents. Each process reads its own chunk.
        with ProcessPoolExecutor(max_workers=len(jobs), mp_context=get_context("spawn")) as pool:
            for counts in pool.map(_pretokenize_chunk, jobs):
                total_counts.update(counts)

    return train_bpe(total_counts, vocab_size, special_tokens)


def _pretokenize_chunk(job) -> dict[str, int]:
    input_path, start, end, special_token_pattern = job
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    parts = re.split(special_token_pattern, chunk) if special_token_pattern else [chunk]
    return pretokenization([part for part in parts if part])


_PATTERN = regex.compile(r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+")


def pretokenization(chunk: list[str]) -> dict[str, int]:
    counts = _core.Counts()
    for part in chunk:
        _core.count_tokens(_PATTERN.findall(part), counts)
    return {entry.first.decode("utf-8"): int(entry.second) for entry in counts}


def train_bpe(counts: dict[str, int], vocab_size: int, special_tokens: list[str]):
    special_tokens = list(dict.fromkeys(special_tokens))
    if any(not token for token in special_tokens):
        raise ValueError("Special tokens must not be empty")
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must fit all 256 bytes and special tokens")
    cpp_counts = _core.Counts()
    for word, count in counts.items():
        cpp_counts[word.encode("utf-8")] = count
    result = _core.train(cpp_counts, vocab_size, special_tokens)
    # BPE tokens may contain incomplete UTF-8; latin1 preserves every byte.
    vocab = {i: token.decode("latin1").encode("latin1") for i, token in enumerate(result.vocab)}
    merges = [
        (pair.first.decode("latin1").encode("latin1"), pair.second.decode("latin1").encode("latin1"))
        for pair in result.merges
    ]
    return vocab, merges
