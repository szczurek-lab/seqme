import numpy as np
import pytest

import seqme as sm

pytest.importorskip("transformers")
pytest.importorskip("einops")


@pytest.fixture(scope="module")
def gena_lm():
    return sm.models.GenaLM(sm.models.GenaLMCheckpoint.bert_base_t2t)


def test_shape_and_means(gena_lm):
    data = [
        "ATGGG",
        "ATGAA",
    ]
    embeddings = gena_lm(data)

    assert embeddings.shape == (2, 768)

    expected_means = np.array(
        [
            2.38051128387,
            2.67395281791,
        ]
    )
    actual_means = embeddings.mean(axis=-1)

    assert actual_means.tolist() == pytest.approx(expected_means.tolist(), abs=1e-6)
