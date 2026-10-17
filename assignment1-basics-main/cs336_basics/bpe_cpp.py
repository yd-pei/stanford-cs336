"""BPE with cppyy: Python handles regex, C++ handles counting and merges."""

import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from itertools import islice
import regex

import cppyy

cppyy.cppdef(r"""
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <unordered_map>
#include <cstdint>

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

Result train(Counts& counts, int vocab_size,
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

    // Store four-byte token IDs instead of a string object for every byte.
    std::unordered_map<std::string, uint32_t> token_ids;
    for (uint32_t i = 0; i < 256; i++) token_ids[result.vocab[i]] = i;
    std::vector<std::vector<uint32_t>> words;
    std::vector<long long> frequencies;
    words.reserve(counts.size());
    frequencies.reserve(counts.size());
    while (!counts.empty()) {
        auto entry = counts.begin();
        const auto& [word, count] = *entry;
        std::vector<uint32_t> tokens;
        tokens.reserve(word.size());
        for (unsigned char byte : word) tokens.push_back(byte);
        words.push_back(std::move(tokens));
        frequencies.push_back(count);
        counts.erase(entry);
    }

    while (static_cast<int>(result.vocab.size()) < vocab_size) {
        // Pack two 32-bit IDs into one key.
        std::unordered_map<uint64_t, long long> pair_counts;
        for (size_t w = 0; w < words.size(); w++)
            for (size_t i = 0; i + 1 < words[w].size(); i++) {
                uint64_t key = (uint64_t(words[w][i]) << 32) | words[w][i + 1];
                pair_counts[key] += frequencies[w];
            }
        if (pair_counts.empty()) break;

        auto as_pair = [&](uint64_t key) {
            return Pair{result.vocab[key >> 32], result.vocab[uint32_t(key)]};
        };
        auto best = pair_counts.begin();
        for (auto it = pair_counts.begin(); it != pair_counts.end(); ++it)
            if (it->second > best->second ||
                (it->second == best->second && as_pair(it->first) > as_pair(best->first)))
                best = it;
        uint32_t left = best->first >> 32, right = uint32_t(best->first);
        Pair pair = as_pair(best->first);
        std::string merged = pair.first + pair.second;
        auto [entry, inserted] = token_ids.emplace(merged, uint32_t(result.vocab.size()));
        uint32_t merged_id = entry->second;
        result.vocab.push_back(merged);
        result.merges.push_back(pair);

        for (auto& tokens : words) {
            size_t out = 0;
            for (size_t i = 0; i < tokens.size(); i++) {
                if (i + 1 < tokens.size() && tokens[i] == left && tokens[i + 1] == right) {
                    tokens[out++] = merged_id;
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
    num_processes: int = 4,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    if chunk_bytes < 1:
        raise ValueError("chunk_bytes must be positive")
    if num_processes < 1:
        raise ValueError("num_processes must be positive")
    if any(token == "" for token in special_tokens):
        raise ValueError("Special tokens must not be empty")

    # Escape regex metacharacters and match longer tokens first when they overlap.
    special_token_pattern = "|".join(
        re.escape(token) for token in sorted(set(special_tokens), key=len, reverse=True)
    )
    total_counts = Counter()
    
    split_token = special_tokens[0].encode("utf-8") if special_tokens else None
    jobs = (
        (str(input_path), start, end, special_token_pattern)
        for start, end in _chunk_ranges(input_path, chunk_bytes, split_token)
    )
    if num_processes == 1:
        for job in jobs:
            total_counts.update(_pretokenize_chunk(job))
    else:
        with ProcessPoolExecutor(max_workers=num_processes, mp_context=get_context("spawn")) as pool:
            # Submit only one batch at a time, so finished results cannot pile up.
            while batch := list(islice(jobs, num_processes)):
                for counts in pool.map(_pretokenize_chunk, batch):
                    total_counts.update(counts)
                    del counts

    return train_bpe(total_counts, vocab_size, special_tokens, consume_counts=True)


def _chunk_ranges(input_path, chunk_bytes, split_token):
    """Choose small chunks ending at document boundaries, including split read buffers."""
    with open(input_path, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        if not split_token:
            # Without a document marker, an arbitrary split could change regex matches.
            if size:
                yield 0, size
            return
        start = 0
        while start < size:
            end = min(start + chunk_bytes, size)
            f.seek(end)
            while end < size:
                data = f.read(64 * 1024 + len(split_token) - 1)
                found = data.find(split_token)
                if found >= 0:
                    end += found
                    break
                if f.tell() == size:
                    end = size
                    break
                end += 64 * 1024
                f.seek(end)
            yield start, end
            start = end


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


def train_bpe(counts: dict[str, int], vocab_size: int, special_tokens: list[str], *, consume_counts=False):
    special_tokens = list(dict.fromkeys(special_tokens))
    if any(not token for token in special_tokens):
        raise ValueError("Special tokens must not be empty")
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must fit all 256 bytes and special tokens")
    cpp_counts = _core.Counts()
    if consume_counts:
        # The top-level pipeline owns this table; release entries during conversion.
        while counts:
            word, count = counts.popitem()
            cpp_counts[word.encode("utf-8")] = count
    else:
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
