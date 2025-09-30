import numpy as np
import pytest

from seqme.models import Hyformer, HyformerCheckpoint

pytest.importorskip("transformers")

_EMBEDDING_DIM = 512


@pytest.fixture(scope="module")
def hyformer():
    return Hyformer(model_name=HyformerCheckpoint.peptides_34M_mic, batch_size=32, device="cpu")


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
    assert actual_means.tolist() == pytest.approx(expected_means.tolist(), abs=1e-6)

    # test perplexity
    perplexity = hyformer.compute_perplexity(data)
    assert perplexity.shape == (len(data),)
    expected_perplexity = [3.03738141, 5.75629187]
    assert perplexity.tolist() == pytest.approx(expected_perplexity, abs=1e-6)

    # test generation
    generated_samples = hyformer.generate(num_samples=2, seed=1337)
    assert isinstance(generated_samples, list)
    assert isinstance(generated_samples[0], str)
    expected_samples = ['WKKIWRRVV', 'RLQNIGKKVVKRLAKKLRRLKKK']
    assert generated_samples == expected_samples

    predictions = hyformer.predict(data)
    assert predictions.shape == (len(data), 1)
    assert isinstance(predictions[0], float)
    expected_predictions = [0.46380734, 1.0693561]
    assert predictions.tolist() == pytest.approx(expected_predictions, abs=1e-6)
