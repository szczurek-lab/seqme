import random
import pytest
from pepme.models.properties import Gravy
from pepme.metrics.conformity_score import ConformityScore


def generate_sequences_from_aas(aa_list, n_seqs, l=30):
    rng = random.Random(0)
    return ["".join(rng.choices(aa_list, k=l)) for _ in range(n_seqs)]


def test_conformity_score_match():
    neg_kd = ["D", "E", "H", "R", "N"]  # AAs with low Kyte-Doolittle index
    sequences = generate_sequences_from_aas(neg_kd, 1100)
    reference = sequences[:1000]
    test = sequences[1000:]

    metric = ConformityScore(reference=reference, descriptors=[Gravy()])
    result = metric(test)

    assert result.value == pytest.approx(0.5, abs=0.05)


def test_conformity_score_mismatch():
    neg_kd = ["D", "E", "H", "R", "N"]  # AAs with low Kyte-Doolittle index
    pos_kd = ["I", "L", "V", "R"]  # AAs with high Kyte-Doolittle index
    reference = generate_sequences_from_aas(neg_kd, 1000)
    test = generate_sequences_from_aas(pos_kd, 100)

    metric = ConformityScore(reference=reference, descriptors=[Gravy()])
    result = metric(test)

    assert result.value == pytest.approx(0.0, abs=0.05)
