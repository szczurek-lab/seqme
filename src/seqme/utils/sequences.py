import random


def shuffle_sequences(sequences: list[str], seed: int = 42) -> list[str]:
    """
    Randomly shuffle characters within each sequence, preserving reproducibility.

    Args:
        sequences: List of input strings to shuffle.
        seed: Seed for the random number generator to ensure determinism.

    Returns:
        A new list where each sequence's characters have been shuffled.
    """
    rng = random.Random(seed)
    shuffled = []
    for seq in sequences:
        chars = list(seq)
        rng.shuffle(chars)
        shuffled.append("".join(chars))
    return shuffled


def random_subset(sequences: list[str], n_samples: int, seed: int = 42) -> list[str]:
    """
    Select a random subset of unique sequences with deterministic behavior.

    Args:
        sequences: The list of input sequences to sample from.
        n_samples: The number of sequences to sample.
        seed: The random seed for reproducibility.

    Returns:
        A list of `n_samples` randomly chosen, unique sequences.

    Raises:
        ValueError: If `n_samples` exceeds the number of available sequences.
    """
    if n_samples > len(sequences):
        raise ValueError(f"Cannot sample {n_samples} sequences from a list of length {len(sequences)}.")

    rng = random.Random(seed)
    return rng.sample(sequences, n_samples)


def read_fasta_file(path: str) -> list[str]:
    """Retrieve sequences from a fasta file."""
    sequences: list[str] = []
    current_seq: list[str] = []

    with open(path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            if line.startswith(">"):
                if current_seq:
                    sequence = "".join(current_seq)
                    if sequence:
                        sequences.append(sequence)
                    current_seq = []
            else:
                current_seq.append(line)

        # Add the last sequence if present
        if current_seq:
            sequence = "".join(current_seq)
            if sequence:
                sequences.append(sequence)

    return sequences


def write_to_fasta_file(sequences: list[str], path: str, headers: list[str] | None = None):
    """Write sequences to a fasta file."""
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            header = headers[i] if headers else f">sequence_{i + 1}"
            f.write(f"{header}\n")
            f.write(f"{seq}\n")
