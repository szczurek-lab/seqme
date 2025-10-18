import random

import pytest

from seqme.metrics import ConformityScore
from seqme.models import Gravy


def generate_sequences_from_aas(aa_list, n_seqs, l=30):
    rng = random.Random(0)
    return ["".join(rng.choices(aa_list, k=l)) for _ in range(n_seqs)]


def test_conformity_score_match():
    neg_kd = ["D", "E", "H", "R", "N"]  # AAs with low Kyte-Doolittle index
    sequences = generate_sequences_from_aas(neg_kd, 1100)
    reference = sequences[:1000]
    test = sequences[1000:]

    metric = ConformityScore(reference=reference, predictors=[Gravy()])

    assert metric.name == "Conformity score"
    assert metric.objective == "maximize"

    result = metric(test)

    assert result.value == pytest.approx(0.5, abs=0.05)


def test_conformity_score_mismatch():
    neg_kd = ["D", "E", "H", "R", "N"]  # AAs with low Kyte-Doolittle index
    pos_kd = ["I", "L", "V", "R"]  # AAs with high Kyte-Doolittle index
    reference = generate_sequences_from_aas(neg_kd, 1000)
    test = generate_sequences_from_aas(pos_kd, 100)

    metric = ConformityScore(reference=reference, predictors=[Gravy()])

    assert metric.name == "Conformity score"
    assert metric.objective == "maximize"

    result = metric(test)

    assert result.value == pytest.approx(0.0, abs=0.05)


def test_one_split():
    neg_kd = ["D", "E", "H", "R", "N"]  # AAs with low Kyte-Doolittle index
    reference = generate_sequences_from_aas(neg_kd, 1000)

    with pytest.raises(ValueError, match=r"Number of cross-validation folds for KDE \(n_splits\) must be at least 2."):
        ConformityScore(reference=reference, predictors=[Gravy()], n_splits=1)
