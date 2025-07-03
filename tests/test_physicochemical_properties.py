import numpy as np
import pytest

from pepme.embeddings.physicochemical_embeddings import PhysicochemicalEmbedding


@pytest.fixture
def sequences():
    return ["AKKLL", "GIGAVLKVLTT", "FFWLL"]


@pytest.mark.parametrize("scaling", ["none", "minmax", "standard"])
def test_embedding_shape_and_type(sequences, scaling):
    model = PhysicochemicalEmbedding(
        properties=["charge", "gravy", "mol_weight"], scaling=scaling
    )
    result = model(sequences)

    # shape: (n_sequences, n_features)
    assert isinstance(result, np.ndarray)
    assert result.shape == (3, 3)


def test_embedding_output_range_minmax(sequences):
    model = PhysicochemicalEmbedding(
        properties=["charge", "gravy", "mol_weight"], scaling="minmax"
    )
    result = model(sequences)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_embedding_output_mean_std(sequences):
    model = PhysicochemicalEmbedding(
        properties=["charge", "gravy", "mol_weight"], scaling="standard"
    )
    result = model(sequences)
    means = np.mean(result, axis=0)
    stds = np.std(result, axis=0)

    # Mean should be close to 0, std close to 1 (for >2 samples)
    assert np.all(np.abs(means) < 1e-5)
    assert np.all(np.abs(stds - 1) < 1e-5)


def test_invalid_scaling_raises(sequences):
    with pytest.raises(ValueError):
        PhysicochemicalEmbedding(properties=["charge", "gravy"], scaling="invalid")(
            sequences
        )


def test_custom_property_subset(sequences):
    model = PhysicochemicalEmbedding(properties=["gravy"], scaling="none")
    result = model(sequences)
    assert result.shape == (3, 1)
