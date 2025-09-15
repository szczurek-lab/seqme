import numpy as np
import pytest

import seqme as sm

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def dna_bert2():
    return sm.models.DNABert2()


def test_dna_bert2_shape_and_means(dna_bert2):
    data = [
        "RVKRVWPLVIRTVIAGYNLYRAIKKK",
        "RKRIHIGPGRAFYTT",
        "DSHAKRHHGYKRKFHEKHHSHRGY",
        "ENREVPPGFTALIKTLRKCKII",
        "NLVSGLIEARKYLEQLHRKLKNCKV",
        "FLPKTLRKFFARIRGGRAAVLNALGKEEQIGRASNSGRKCARKKK",
    ]
    embeddings = dna_bert2(data)

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
