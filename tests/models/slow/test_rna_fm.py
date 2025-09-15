import numpy as np
import pytest

import seqme as sm

pytest.importorskip("rna-fm")


@pytest.fixture(scope="module")
def rna_fm():
    return sm.models.DNABert2()


def test_rna_fm_shape_and_means(rna_fm):
    data = [
        "RVKRVWPLVIRTVIAGYNLYRAIKKK",
        "RKRIHIGPGRAFYTT",
        "DSHAKRHHGYKRKFHEKHHSHRGY",
        "ENREVPPGFTALIKTLRKCKII",
        "NLVSGLIEARKYLEQLHRKLKNCKV",
        "FLPKTLRKFFARIRGGRAAVLNALGKEEQIGRASNSGRKCARKKK",
    ]
    embeddings = rna_fm(data)

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
