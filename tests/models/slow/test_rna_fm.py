import numpy as np
import pytest

import seqme as sm

pytest.importorskip("fm")


@pytest.fixture(scope="module")
def rna_fm():
    return sm.models.RNAFM()


def test_rna_fm_shape_and_means(rna_fm):
    data = [
        "AAAUUU",
        "UUU",
    ]
    embeddings = rna_fm(data)

    assert embeddings.shape == (2, 1280)

    expected_means = np.array(
        [
            0.023362776,
            0.000433295,
        ]
    )
    actual_means = embeddings.mean(axis=-1)

    assert actual_means.tolist() == pytest.approx(expected_means.tolist(), abs=1e-6)
