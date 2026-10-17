import pickle
from pathlib import Path
from time import perf_counter

from cs336_basics.bpe_cpp import bpe_cpp as bpe


if __name__ == "__main__":
    start = perf_counter()
    vocab, merges = bpe(
        input_path="data/owt_train.txt",
        vocab_size=32_000,
        special_tokens=["<|endoftext|>"],
    )
    print(f"Training time: {perf_counter() - start:.2f} seconds")

    output_dir = Path("data/bpe_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "vocab.pkl").open("wb") as f:
        pickle.dump(vocab, f)
    with (output_dir / "merges.pkl").open("wb") as f:
        pickle.dump(merges, f)
