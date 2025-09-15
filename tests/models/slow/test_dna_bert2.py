import numpy as np
import pytest

import seqme as sm

pytest.importorskip("transformers")
pytest.importorskip("einops")


@pytest.fixture(scope="module")
def dna_bert2():
    return sm.models.DNABert2()


def test_dna_bert2_shape_and_means(dna_bert2):
    data = [
        "RVKRVWPLVIRTVIAGYNLYRAIKKK",
        "RKRIHIGPGRAFYTT",
    ]
    embeddings = dna_bert2(data)

    assert embeddings.shape == (2, 768)

    expected_means = np.array(
        [
            0.00342898,
            0.00424373,
        ]
    )
    actual_means = embeddings.mean(axis=-1)

    assert actual_means.tolist() == pytest.approx(expected_means.tolist(), abs=1e-6)
