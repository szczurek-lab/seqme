import numpy as np
import pytest

from seqme.metrics import Precision, Recall


def test_basic_precision():
    reference = ["A" * 15, "A" * 17]
    metric = Precision(
        reference=reference,
        embedder=length_mock_embedder,
        n_neighbors=1,
        batch_size=1,
    )

    assert metric.name == "Precision"
    assert metric.objective == "maximize"

    result = metric(["A" * 2, "A" * 16])
    assert result.value == 0.5


def test_basic_recall():
    reference = ["A" * 15, "A" * 17]
    metric = Recall(
        reference=reference,
        embedder=length_mock_embedder,
        n_neighbors=1,
        batch_size=1,
    )

    assert metric.name == "Recall"
    assert metric.objective == "maximize"

    result = metric(["A" * 2, "A" * 16])
    assert result.value == 1.0


def test_precision_with_larger_neighborhood():
    reference = ["A" * 15, "A" * 17, "A" * 1]
    metric = Precision(
        reference=reference,
        embedder=length_mock_embedder,
        n_neighbors=2,
        batch_size=1,
        strict=False,
    )

    assert metric.name == "Precision"
    assert metric.objective == "maximize"

    result = metric(["A" * 33, "A" * 34])
    assert result.value == 0.5


def test_identical_sequences_precision():
    reference = ["KKAA", "KKAA", "KKKA", "KKAK"]
    metric = Precision(
        n_neighbors=3,
        reference=reference,
        embedder=mock_embedder,
    )

    result = metric(sequences=reference)
    assert result.value == 1.0


def test_identical_sequences_recall():
    reference = ["KKAA", "KKAA", "KKKA", "KKAK"]
    metric = Recall(
        n_neighbors=3,
        reference=reference,
        embedder=mock_embedder,
    )

    result = metric(sequences=reference)
    assert result.value == 1.0


def test_empty_reference():
    reference = []
    with pytest.raises(ValueError):
        metric = Precision(
            n_neighbors=1,
            reference=reference,
            embedder=mock_embedder,
        )
        metric(sequences=["KKAA", "KKAA"])


def test_empty_sequences():
    reference = ["KKAA", "KKAA"]
    metric = Precision(
        n_neighbors=1,
        reference=reference,
        embedder=mock_embedder,
    )

    with pytest.raises(ValueError):
        metric(sequences=[])


def test_precision_recall():
    reference = [
        "LVFEKKLKKTLR",
        "MSQTLLPLYAANHVTKFEMYQSSGYR",
        "VKKEAKKKLEERL",
        "GLPVIRGKCITKKGLKI",
        "VRSKKILEFGAKLSVRYLETVATGWKRT",
        "MFHALPAAAACQRHI",
        "TGVALSADNLFELAEKDKIIKEI",
        "FLTILLLGAVNSV",
        "HGALIFRRRLPKIAWGGKKFF",
        "MVELVRLEHTRKQMIHLSGFTLFCMAQINKYT",
    ]
    sequences = [
        "MLWKRRSEIILKGGARSSKILLEGAAQTK",
        "QSLLLPDDAAKVV",
        "LRAKRIFDIFLV",
        "MYCLRIIKIGGVGSSKQLLCLDAIAVVIVIES",
        "MLTLDRLFVINKEGIYCSDCRLFHIAPI",
        "MIQCHDLVKSARRLVT",
        "KFTFELMKVANVRKKIIHDC",
        "RPCKIWKKLSCL",
        "WRCEVILKKWWRLQN",
        "ITYAGMAVFSTPLPEMAAYTVKIPELID",
    ]
    metric_precision = Precision(
        reference=reference,
        embedder=aa_embedder,
        n_neighbors=1,
    )

    metric_recall = Recall(
        reference=reference,
        embedder=aa_embedder,
        n_neighbors=1,
    )

    precision = metric_precision(sequences=sequences)
    recall = metric_recall(sequences=sequences)

    assert precision.value == 0.7
    assert recall.value == 0.8


# -------------------- Embedders --------------------


def length_mock_embedder(sequences: list[str]) -> np.ndarray:
    lengths = [len(sequence) for sequence in sequences]
    return np.array(lengths).reshape(-1, 1).astype(np.float64)


def mock_embedder(seqs: list[str]) -> np.ndarray:
    n_ks = [seq.count("K") for seq in seqs]
    zeros = [0] * len(seqs)
    return np.array(list(zip(n_ks, zeros, strict=True))).astype(np.float64)


def aa_embedder(seqs: list[str]) -> np.ndarray:
    aa_to_int = {
        "A": 0,
        "C": 1,
        "D": 2,
        "E": 3,
        "F": 4,
        "G": 5,
        "H": 6,
        "I": 7,
        "K": 8,
        "L": 9,
        "M": 10,
        "N": 11,
        "P": 12,
        "Q": 13,
        "R": 14,
        "S": 15,
        "T": 16,
        "V": 17,
        "W": 18,
        "Y": 19,
        "X": 20,
    }

    max_len = max(len(seq) for seq in seqs)
    batch_size = len(seqs)
    arr = np.full((batch_size, max_len), 21, dtype=np.int32)  # 21 = PAD

    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq):
            arr[i, j] = aa_to_int.get(aa.upper(), aa_to_int["X"])
    return arr.astype(np.float64)
