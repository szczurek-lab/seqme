import numpy as np
import pytest

import seqme as sm

pytest.importorskip("transformers")
pytest.importorskip("einops")


@pytest.fixture(scope="module")
def gena_lm_embedder():
    return sm.models.GENALM(sm.models.GENALMCheckpoint.bert_base_t2t)


@pytest.fixture(scope="module")
def gena_lm_promoter():
    return sm.models.GENALM(sm.models.GENALMCheckpoint.bert_base_t2t_promoters)


@pytest.fixture(scope="module")
def gena_lm_splicing():
    return sm.models.GENALM(sm.models.GENALMCheckpoint.bert_base_t2t_splice_site)


def test_embedder_shape_and_means(gena_lm_embedder):
    sequences = ["ATGGG", "ATGAA"]
    embeddings = gena_lm_embedder(sequences)

    assert embeddings.shape == (2, 768)

    expected_means = np.array(
        [
            2.38051128387,
            2.67395281791,
        ]
    )
    actual_means = embeddings.mean(axis=-1)

    assert actual_means.tolist() == pytest.approx(expected_means.tolist(), abs=1e-6)


def test_promoter_shape_and_means(gena_lm_promoter):
    sequences = [
        "ATGGG",
        "ATGAA",
    ]
    prediction = gena_lm_promoter(sequences)

    assert prediction.shape == (2, 2)

    expected_sums = np.array([1.0, 1.0])
    actual_sums = prediction.sum(axis=-1)

    assert actual_sums.tolist() == pytest.approx(expected_sums.tolist(), abs=1e-6)


def test_splicing_site_shape_and_means(gena_lm_splicing):
    sequences = [
        "ATGGG",
        "ATGAA",
    ]
    prediction = gena_lm_splicing(sequences)

    assert prediction.shape == (2, 3)

    expected_sums = np.array([1.0, 1.0])
    actual_sums = prediction.sum(axis=-1)

    assert actual_sums.tolist() == pytest.approx(expected_sums.tolist(), abs=1e-6)
