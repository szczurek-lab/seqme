import numpy as np
import pytest

from seqme.models import ESMIF1

pytest.importorskip("fair-esm")
pytest.importorskip("torch-scatter")


@pytest.fixture(scope="module")
def esm_if():
    return ESMIF1(device="cpu")


def test_esm_if_perplexity(esm_if):
    sequences = ["KKKK", "RR"]

    lens = [len(seq) for seq in sequences]
    coords = [np.ones((seq_len, 3, 3), dtype=np.float32) for seq_len in lens]

    perplexity = esm_if.compute_perplexity(coords, sequences)

    assert perplexity.shape == (2,)

    expected_means = np.array([21.530224, 30.538822])

    assert perplexity.tolist() == pytest.approx(expected_means.tolist(), abs=1e-6)
