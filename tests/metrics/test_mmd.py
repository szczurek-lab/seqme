import numpy as np
import pytest

from seqme.metrics import MMD


def embedder(seqs: list[str]) -> np.ndarray:
    n_ks = [seq.count("K") for seq in seqs]
    zeros = [0] * len(seqs)
    return np.array(list(zip(n_ks, zeros, strict=True)))


def test_shifted():
    reference = ["KKAA", "KKAA"]
    metric = MMD(
        reference=reference,
        embedder=embedder,
        reference_name="Random",
        embedder_name="embedder",
    )

    assert metric.name == "MMD@embedder (Random)"
    assert metric.objective == "minimize"

    result = metric(["KAAA", "KAAA"])
    assert result.value == pytest.approx(9.975075721740723, abs=1e-7)


def test_the_same():
    reference = ["KKAA", "KKAA"]
    metric = MMD(reference=reference, embedder=embedder)
    result = metric(["KKAA", "KKAA"])
    assert result.value == 0.0


def test_empty_reference():
    with pytest.raises(ValueError):
        MMD(reference=[], embedder=embedder)


def test_empty_sequences():
    reference = ["KKAA", "KKAA"]
    metric = MMD(reference=reference, embedder=embedder)
    with pytest.raises(ValueError):
        metric([])
