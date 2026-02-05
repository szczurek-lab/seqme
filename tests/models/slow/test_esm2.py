import numpy as np
import pytest

from seqme.models import ESM2, ESM2Checkpoint

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def esm():
    return ESM2(model_name=ESM2Checkpoint.t6_8M, batch_size=32, device="cpu")


def test_esm2_shape_and_means(esm):
    sequences = [
        "RVKRVWPLVIRTVIAGYNLYRAIKKK",
        "RKRIHIGPGRAFYTT",
        "DSHAKRHHGYKRKFHEKHHSHRGY",
        "ENREVPPGFTALIKTLRKCKII",
        "NLVSGLIEARKYLEQLHRKLKNCKV",
        "FLPKTLRKFFARIRGGRAAVLNALGKEEQIGRASNSGRKCARKKK",
    ]
    embeddings = esm(sequences)

    assert embeddings.shape == (6, 320)

    expected_means = np.array([-0.01061969, -0.01052918, -0.01140676, -0.00957893, -0.00982053, -0.0104174])
    actual_means = embeddings.mean(axis=-1)

    assert actual_means.tolist() == pytest.approx(expected_means.tolist(), abs=1e-6)


def test_pseudo_perplexity(esm):
    sequences = ["MMRK", "RKSPL", "RRLSK", "RRLSK"]

    actual_ppls = esm.compute_pseudo_perplexity(sequences)
    assert actual_ppls.shape == (4,)
    expected_ppls = np.array([4.6984897, 8.919382, 8.295634, 8.295634])
    assert actual_ppls.tolist() == pytest.approx(expected_ppls.tolist(), abs=1e-6)

    actual_ppls2 = esm.compute_pseudo_perplexity(sequences, mask_size=2)
    assert actual_ppls2.shape == (4,)
    expected_ppls2 = np.array([4.558269, 8.614358, 7.384721, 7.384721])
    assert actual_ppls2.tolist() == pytest.approx(expected_ppls2.tolist(), abs=1e-6)
