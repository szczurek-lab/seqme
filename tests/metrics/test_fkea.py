import numpy as np
import pytest

from seqme.metrics import FKEA


@pytest.fixture
def shifted_embedder():
    def _embedder(seqs: list[str]) -> np.ndarray:
        # count Ks, pad with zeros in other dims
        n_ks = [seq.count("K") for seq in seqs]
        zeros = [0] * len(seqs)
        return np.array(list(zip(n_ks, zeros, strict=True)), dtype=np.float32)

    return _embedder


def test_fkea_overlap(shifted_embedder):
    metric = FKEA(
        embedder=shifted_embedder,
        bandwidth=2.0,
        n_random_fourier_features=256,
        alpha=2.0,
        seed=42,
    )

    assert metric.name == "FKEA"
    assert metric.objective == "maximize"

    result = metric(["KAAA"] * 10)
    assert pytest.approx(result.value) == 1.0
    assert result.deviation is None


def test_fkea_two_modes(shifted_embedder):
    metric = FKEA(embedder=shifted_embedder, bandwidth=10.0, n_random_fourier_features=2, alpha=2.0, seed=42)

    assert metric.name == "FKEA"
    assert metric.objective == "maximize"

    result = metric(["KAAA", "RRRRRRRRRR", "RRRRRRRRRRR"])
    assert result.value == pytest.approx(1.0057, abs=1e-4)
    assert result.deviation is None

    with pytest.raises(ValueError, match=r"^Computed the metric using different number of sequences.$"):
        metric(["KAAA", "RRRRRRRRRR"])


def test_fkea_different_lengths(shifted_embedder):
    metric = FKEA(
        embedder=shifted_embedder,
        bandwidth=10.0,
        n_random_fourier_features=2,
        alpha=2.0,
    )

    assert metric.name == "FKEA"
    assert metric.objective == "maximize"

    result = metric(["KAAA", "RRRRRRRRRR", "RRRRRRRRRRR"])
    assert pytest.approx(result.value) == pytest.approx(1.1099, abs=1e-4)
    assert result.deviation is None


def test_vendi_different_lengths(shifted_embedder):
    metric = FKEA(
        embedder=shifted_embedder,
        bandwidth=10.0,
        n_random_fourier_features=None,
        alpha=2.0,
        seed=42,
    )

    assert metric.name == "FKEA"
    assert metric.objective == "maximize"

    result = metric(["KAAA", "RRRRRRRRRR", "RRRRRRRRRRR"])
    assert pytest.approx(result.value) == pytest.approx(1.0876, abs=1e-4)
    assert result.deviation is None


def test_invalid_alpha(shifted_embedder):
    with pytest.raises(ValueError, match=r"^Expected alpha > 0.$"):
        FKEA(embedder=shifted_embedder, bandwidth=2.0, n_random_fourier_features=32, alpha=0.0)


@pytest.fixture
def two_mode_embedder():
    base_seed = 42
    rng = np.random.default_rng(base_seed)

    dim = 8

    def _embedder(seqs: list[str]) -> np.ndarray:
        rows = []
        for s in seqs:
            mean = 0.0 if "K" in s else 10.0
            rows.append(rng.normal(loc=mean, scale=1.0, size=dim).astype(np.float32))
        return np.vstack(rows)

    return _embedder


def test_fkea_many(two_mode_embedder):
    metric = FKEA(
        embedder=two_mode_embedder,
        bandwidth=2.0,
        n_random_fourier_features=256,
        alpha=2.0,
        seed=42,
    )

    assert metric.name == "FKEA"
    assert metric.objective == "maximize"

    result = metric(["K"] * 200 + ["A"] * 200)
    assert pytest.approx(result.value) == 186.262786
    assert result.deviation is None

    result = metric(["K"] * 400 + ["A"] * 0)
    assert pytest.approx(result.value) == 165.7301025
    assert result.deviation is None
