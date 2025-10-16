import random

import pytest

from seqme.metrics import KLDivergence
from seqme.models import Gravy


def generate_sequences_from_aas(aa_list, n_seqs, l=30):
    rng = random.Random(0)
    return ["".join(rng.choices(aa_list, k=l)) for _ in range(n_seqs)]


def test_same_sequences():
    neg_kd = ["D", "E", "H", "R", "N"]  # AAs with low Kyte-Doolittle index
    sequences = generate_sequences_from_aas(neg_kd, 1100)
    reference = sequences[:1000]
    test = sequences[1000:]

    metric = KLDivergence(
        reference=reference,
        predictor=Gravy(),
        n_draws=1_000,
        kde_bandwidth=0.2,
    )

    assert metric.name == "KL-divergence"
    assert metric.objective == "minimize"

    result = metric(test)

    assert result.value == pytest.approx(0, abs=0.001)
    assert result.deviation == pytest.approx(0, abs=0.001)


def test_different_sequences():
    neg_kd = ["D", "E", "H", "R", "N"]  # AAs with low Kyte-Doolittle index
    pos_kd = ["I", "L", "V", "R"]  # AAs with high Kyte-Doolittle index
    reference = generate_sequences_from_aas(neg_kd, 1000)
    test = generate_sequences_from_aas(pos_kd, 100)

    metric = KLDivergence(reference=reference, predictor=Gravy())

    assert metric.name == "KL-divergence"
    assert metric.objective == "minimize"

    result = metric(test)

    assert result.value == pytest.approx(90.7, abs=0.01)
    assert result.deviation == pytest.approx(0.13, abs=0.01)
