import numpy as np
from seqme.metrics.mutational_robustness import MutationalRobustness, compute_mutational_robustness


def fake_predictor(seq_list):
    # Simple length-based dummy predictor
    return np.array([len(s) for s in seq_list])


def test_mutational_robustness_metric():
    metric = MutationalRobustness(predictor=fake_predictor, seed=42)

    # Check metadata
    assert metric.name == "MutationalRobustness"
    assert metric.objective == "maximize"

    # Compute metric
    result = metric(["MKQW", "RKSPL", "MMRK"])
    assert np.isfinite(result.value)
    assert result.deviation is None


def test_compute_function_matches_class():
    seqs = ["MKQW", "RKSPL", "MMRK"]
    val1 = compute_mutational_robustness(seqs, predictor=fake_predictor)
    val2 = MutationalRobustness(predictor=fake_predictor)(seqs).value
    assert np.isclose(val1, val2)
