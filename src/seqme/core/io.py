import pickle
from pathlib import Path
from typing import Any


def read_fasta(path: str | Path) -> list[str]:
    """Retrieve sequences from a FASTA file.

    Args:
        path: Path to FASTA file.

    Returns:
        The list of sequences.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    sequences: list[str] = []
    current_seq: list[str] = []

    with path.open() as f:
        for line in f:
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


def to_fasta(sequences: list[str], path: str | Path, *, headers: list[str] | None = None):
    """Write sequences to a FASTA file.

    Args:
       sequences: List of text sequences.
       path: Output filepath, e.g., ``"/path/seqs.fasta"``.
       headers: Optional sequence names.
    """
    if headers is not None and len(headers) != len(sequences):
        raise ValueError("headers length must match sequences length")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        for i, seq in enumerate(sequences):
            header = headers[i] if headers else f">seq_{i + 1}"

            if not header.startswith(">"):
                header = ">" + header

            f.write(f"{header}\n")
            f.write(f"{seq}\n")


def read_pickle(path: str | Path) -> Any:
    """Load and return an object from a pickle file.

    Args:
        path: Path to pickle file.

    Returns:
        The deserialized Python object.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    with path.open("rb") as f:
        return pickle.load(f)


def to_pickle(content: Any, path: str | Path):
    """Serialize an object and write it to a pickle file.

    Args:
       content: Pickable object.
       path: Output filepath, e.g., ``"/path/cache.pkl"``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("wb") as f:
        pickle.dump(content, f)
