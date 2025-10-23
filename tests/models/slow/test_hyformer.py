import numpy as np
import pytest

from seqme.models import Hyformer, HyformerCheckpoint

pytest.importorskip("hyformer")

_EMBEDDING_DIM = 512
_ABS_TOLERANCE = 1e-3


@pytest.fixture(scope="module")
def hyformer():
    return Hyformer(model_name=HyformerCheckpoint.peptides_34M_mic, device="cpu")


def test_hyformer_shape_and_means(hyformer):
    data = [
        "RVKRVWPLVIRTVIAGYNLYRAIKKK",
        "RKRIHIGPGRAFYTT",
    ]

    # test embeddings
    embeddings = hyformer(data)
    assert embeddings.shape == (len(data), _EMBEDDING_DIM)
    expected_means = np.array([0.03424069, 0.04243201])
    actual_means = embeddings.mean(axis=-1)
    assert actual_means.tolist() == pytest.approx(expected_means.tolist(), abs=_ABS_TOLERANCE)

    # test perplexity
    perplexity = hyformer.compute_perplexity(data)
    assert perplexity.shape == (len(data),)
    expected_perplexity = np.array([3.03738117, 5.75628996])
    assert perplexity.tolist() == pytest.approx(expected_perplexity, abs=_ABS_TOLERANCE)

    # test generation
    generated_samples = hyformer.generate(num_samples=2, seed=1337)
    assert isinstance(generated_samples, list)
    assert isinstance(generated_samples[0], str)
    expected_samples = np.array(["KCKKWKWKKKLV", "RWWRWWRWG"])
    assert len(generated_samples) == len(expected_samples)
    assert (generated_samples == expected_samples).all()

    # test predictions
    predictions = hyformer.predict(data)
    assert predictions.shape == (len(data), 1)
    expected_predictions = np.array([[0.46380645], [1.0693568]])
    assert predictions.tolist() == pytest.approx(expected_predictions, abs=_ABS_TOLERANCE)
