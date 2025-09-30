import numpy as np
import pytest

from seqme.models import Hyformer, HyformerCheckpoint

pytest.importorskip("transformers")

_EMBEDDING_DIM = 512


@pytest.fixture(scope="module")
def hyformer():
    return Hyformer(model_name=HyformerCheckpoint.peptides, batch_size=32, device="cpu")


def test_hyformer_shape_and_means(hyformer):
    data = [
        "RVKRVWPLVIRTVIAGYNLYRAIKKK",
        "RKRIHIGPGRAFYTT",
        "DSHAKRHHGYKRKFHEKHHSHRGY",
        "ENREVPPGFTALIKTLRKCKII",
        "NLVSGLIEARKYLEQLHRKLKNCKV",
        "FLPKTLRKFFARIRGGRAAVLNALGKEEQIGRASNSGRKCARKKK",
    ]
    embeddings = hyformer(data)

    assert embeddings.shape == (6, _EMBEDDING_DIM)

    expected_means = np.array(
        [
            0.00181393,
            0.0472128,
            -0.00306486,
            0.00684623,
            -0.0189725,
            -0.05783607,
        ]
    )
    actual_means = embeddings.mean(axis=-1)

    assert actual_means.tolist() == pytest.approx(expected_means.tolist(), abs=1e-6)

    perplexity = hyformer.compute_perplexity(data)
    assert perplexity.shape == (6,)

    expected_perplexity = [
        7.81769562,
        7.04722548,
        7.89776897,
        22.06753731,
        17.13210678,
        5.95252991,
    ]
    assert perplexity.tolist() == pytest.approx(expected_perplexity, abs=1e-6)

    generated_samples = hyformer.generate(num_samples=2)
    assert isinstance(generated_samples, list)
    assert isinstance(generated_samples[0], str)


    predictions = hyformer.predict(data)
    assert predictions.shape == (6, 1)
    assert isinstance(predictions[0], float)
