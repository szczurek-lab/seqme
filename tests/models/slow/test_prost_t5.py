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
        "DSHAKRHHGYKRKFHEKHHSHRGY",
        "ENREVPPGFTALIKTLRKCKII",
        "NLVSGLIEARKYLEQLHRKLKNCKV",
        "FLPKTLRKFFARIRGGRAAVLNALGKEEQIGRASNSGRKCARKKK",
    ]
    embeddings = prost_t5(data)

    assert embeddings.shape == (6, 320)

    expected_means = np.array(
        [
            -0.01061969,
            -0.01052918,
            -0.01140676,
            -0.00957893,
            -0.00982053,
            -0.0104174,
        ]
    )
    actual_means = embeddings.mean(axis=-1)

    assert actual_means.tolist() == pytest.approx(expected_means.tolist(), abs=1e-6)
