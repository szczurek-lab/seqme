import random

import numpy as np


def shuffle_characters(sequences: list[str], seed: int | None = 0) -> list[str]:
    """
    Randomly shuffle characters within each sequence.

    Args:
        sequences: Sequences to shuffle.
        seed: Local seed when sampling. If ``None``, no fixed local seed is used.

    Returns:
        A new list where each sequences characters have been shuffled.
    """
    rng = random.Random(seed)
    shuffled = []
    for seq in sequences:
        chars = list(seq)
        rng.shuffle(chars)
        shuffled.append("".join(chars))
    return shuffled


def subsample(
    sequences: list[str],
    n_samples: int,
    *,
    return_indices: bool = False,
    seed: int | None = 0,
) -> list[str] | tuple[list[str], np.ndarray]:
    """
    Sample a subset of the sequences without replacement.

    Args:
        sequences: Sequences to sample from.
        n_samples: Number of sequences to sample.
        return_indices: If ``True``, return a tuple of the sequence subset and indices else return only the sequence subset.
        seed: Local seed when sampling. If ``None``, no fixed local seed is used.

    Returns:
        A list of ``n_samples`` randomly chosen sequences. Optionally, including the indices.

    Raises:
        ValueError: If ``n_samples`` exceeds the number of available sequences.
    """
    if n_samples > len(sequences):
        raise ValueError(f"Cannot sample {n_samples} sequences from a list of length {len(sequences)}.")

    rng = np.random.default_rng(seed)
    indices = rng.choice(np.arange(len(sequences), dtype=int), size=n_samples, replace=False)
    subset = [sequences[idx] for idx in indices]

    if return_indices:
        return subset, indices

    return subset
