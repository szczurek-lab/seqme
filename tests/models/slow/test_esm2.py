import numpy as np
import pytest

from seqme.models import ESM2, ESM2Checkpoint

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def esm():
    return ESM2(model_name=ESM2Checkpoint.t6_8M, batch_size=32, device="cpu")


def test_esm2_shape_and_means(esm):
    data = [
        "RVKRVWPLVIRTVIAGYNLYRAIKKK",
        "RKRIHIGPGRAFYTT",
        "DSHAKRHHGYKRKFHEKHHSHRGY",
        "ENREVPPGFTALIKTLRKCKII",
        "NLVSGLIEARKYLEQLHRKLKNCKV",
        "FLPKTLRKFFARIRGGRAAVLNALGKEEQIGRASNSGRKCARKKK",
    ]
    embeddings = esm(data)

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
