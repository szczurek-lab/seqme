import pytest

from seqme.metrics import KmerJaccardSimilarity


def test_name_and_objective_default():
    reference = ["ABC"]
    metric = KmerJaccardSimilarity(reference=reference, n=3)
    assert metric.name == "Jaccard-3"
    assert metric.objective == "minimize"


def test_name_with_reference_name_and_objective_override():
    reference = ["ABC"]
    metric = KmerJaccardSimilarity(reference=reference, n=3, objective="maximize", reference_name="SampleRef")
    assert metric.name == "Jaccard-3 (SampleRef)"
    assert metric.objective == "maximize"


def test_compute_metric_basic():
    # reference 2-grams: {"AB","BC","CD"}, sequences ["ABC","XYZ"]
    # J("ABC") = 2/3, J("XYZ") = 0/5 → avg ≈ 1/3
    reference = ["ABC", "BCD"]
    metric = KmerJaccardSimilarity(reference=reference, n=2)
    result = metric(["ABC", "XYZ"])
    assert result.value == pytest.approx(1 / 3)
    assert result.deviation is None


def test_perfect_overlap():
    reference = ["HELLO"]
    metric = KmerJaccardSimilarity(reference=reference, n=3)
    result = metric(["HELLO"])
    assert result.value == pytest.approx(1.0)
    assert result.deviation is None


def test_sequences_shorter_than_n():
    # refs length < n → ref_ngrams empty; sequences too short → both unions empty
    # similarity treated as 0 each → avg = 0
    reference = ["A", "B"]
    metric = KmerJaccardSimilarity(reference=reference, n=2)
    result = metric(["A", "B"])
    assert result.value == 0.0
    assert result.deviation is None


def test_empty_sequences():
    reference = ["ABC"]
    metric = KmerJaccardSimilarity(reference=reference, n=3)
    result = metric([])
    assert result.value == 0.0
    assert result.deviation is None
