from collections import Counter

import pytest

from cs336_basics.bpe import bpe, pretokenization as python_pre, train_bpe as python_train
from cs336_basics.bpe_cpp import _chunk_ranges, bpe_cpp, pretokenization, train_bpe


@pytest.mark.parametrize('num_processes', [1, 2])
def test_small_chunks_match_python(tmp_path, num_processes):
    path = tmp_path / 'stories.txt'
    text = '<|endoftext|>'.join([
        "Hello, 世界! café १२३ Ⅷ ²\n\t I'm sure we'll go. 😀",
        'Once upon a time, there was a cat. 猫\n',
        'A <s> small story with another boundary.\n',
    ] * 5)
    path.write_text(text, encoding='utf-8')
    specials = ['<|endoftext|>', '<s>']
    assert bpe_cpp(path, 350, specials, num_processes, chunk_bytes=31) == bpe(path, 350, specials)


def test_chunk_boundary_crosses_read_buffer(tmp_path):
    marker = b'<|endoftext|>'
    path = tmp_path / 'boundary.txt'
    data = b'x' * (32 + 65536 - 5) + marker + b'y' * 100
    path.write_bytes(data)
    ranges = list(_chunk_ranges(path, 32, marker))
    assert ranges[0] == (0, data.index(marker))
    assert b''.join(data[start:end] for start, end in ranges) == data
    assert all(data[start:].startswith(marker) for start, _ in ranges[1:])


def test_pretokenization_and_training(tmp_path):
    parts = ['\x00\xff 猫 café Ⅷ ² 😀', "we'll go\n  ", 'abab aaaa ababa']
    counts = pretokenization(parts)
    assert counts == python_pre(parts)
    expected = python_train(counts, 300, ['x', 'x', '<s>'])
    assert train_bpe(counts, 300, ['x', 'x', '<s>']) == expected
    assert counts  # The public API preserves the caller's counts by default.
    owned_counts = Counter(counts)
    assert train_bpe(owned_counts, 300, ['x', 'x', '<s>'], consume_counts=True) == expected
    assert not owned_counts
    path = tmp_path / 'empty.txt'
    path.write_bytes(b'')
    assert bpe_cpp(path, 300, ['<s>'], num_processes=2) == bpe(path, 300, ['<s>'])
    path.write_text('no document marker: 猫', encoding='utf-8')
    assert bpe_cpp(path, 300, [], num_processes=2) == bpe(path, 300, [])
