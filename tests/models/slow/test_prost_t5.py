import numpy as np
import pytest

import seqme as sm

pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")


@pytest.fixture(scope="module")
def prost_t5():
    return sm.models.ProstT5()


def test_prost_t5_shape_and_means(prost_t5):
    data = [
        "RVKRVWPLVIRTVIAGYNLYRAIKKK",
        "RKRIHIGPGRAFYTT",
    ]
    embeddings = prost_t5(data)

    assert embeddings.shape == (2, 1024)

    expected_means = np.array(
        [
            -0.003066932782530,
            6.7310757003724e-06,
        ]
    )
    actual_means = embeddings.mean(axis=-1)

    assert actual_means.tolist() == pytest.approx(expected_means.tolist(), abs=1e-6)
