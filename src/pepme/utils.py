from typing import Optional


def read_fasta_file(path: str) -> list[str]:
    sequences: list[str] = []
    current_seq: list[str] = []

    with open(path, "r") as file:
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


def write_to_fasta_file(
    sequences: list[str],
    path: str,
    headers: Optional[list[str]] = None,
):
    with open(path, "w") as f:
        for i, seq in enumerate(sequences):
            header = headers[i] if headers else f">sequence_{i + 1}"
            f.write(f"{header}\n")
            f.write(f"{seq}\n")
