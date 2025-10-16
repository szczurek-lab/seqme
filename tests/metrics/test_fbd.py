import math

import numpy as np
import pytest

from seqme.metrics import FBD


@pytest.fixture
def shifted_embedder():
    def _embedder(seqs: list[str]) -> np.ndarray:
        # count Ks, pad with zeros in other dims
        n_ks = [seq.count("K") for seq in seqs]
        zeros = [0] * len(seqs)
        return np.array(list(zip(n_ks, zeros, strict=True)))

    return _embedder


def test_shifted_fbd(shifted_embedder):
    reference = ["KKAA", "KKAA"]
    metric = FBD(reference=reference, embedder=shifted_embedder)

    # Name and objective properties
    assert metric.name == "FBD"
    assert metric.objective == "minimize"

    result = metric(["KAAA", "KAAA"])
    assert result.value == 1.0


@pytest.fixture(params=[3, 5])
def zero_embedder(request):
    # test with different embedding dims if you like
    n_dim = request.param

    def embedder(seqs: list[str]) -> np.ndarray:
        return np.zeros((len(seqs), n_dim))

    return embedder


def test_overlapping_fbd(zero_embedder):
    reference = ["KRQS", "AAAA"]
    metric = FBD(reference=reference, embedder=zero_embedder)

    # Name and objective properties
    assert metric.name == "FBD"
    assert metric.objective == "minimize"

    result = metric(["KA", "BBBB"])
    assert result.value == 0.0


def test_single_sequence_nan(zero_embedder):
    reference = ["KRQS", "AAA"]
    metric = FBD(reference=reference, embedder=zero_embedder)

    # Name and objective properties stay the same
    assert metric.name == "FBD"
    assert metric.objective == "minimize"

    result = metric(["KA"])
    assert math.isnan(result.value)
