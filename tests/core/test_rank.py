import numpy as np

import seqme as sm


def test_basic():
    sequences = {
        "model1": ["A"],
        "model2": ["AA", "BB", "CC"],
        "model3": ["AAA", "BBB", "CCC"],
        "model4": ["AA", "BB", "CC", "DDD"],
    }
    metrics = [
        sm.metrics.Count(),
        sm.metrics.Length(objective="maximize"),
    ]

    df = sm.evaluate(sequences, metrics)
    df = sm.rank(df)

    assert df.shape == (4, 3 * 2)
    assert df.attrs["objective"]["Rank"] == "minimize"

    assert np.all(df["Rank"]["value"] == np.array([3, 2, 1, 1]))


def test_mean_rank():
    sequences = {
        "model1": ["A"],
        "model2": ["AA", "BB", "CC"],
        "model3": ["AAA", "BBB", "CCC"],
        "model4": ["AA", "BB", "CC", "DDD"],
    }
    metrics = [
        sm.metrics.Count(),
        sm.metrics.Length(objective="maximize"),
    ]

    df = sm.evaluate(sequences, metrics)
    df = sm.rank(df, tiebreak="mean-rank", name="Rank (mean-rank)")

    assert df.shape == (4, 3 * 2)
    assert df.attrs["objective"]["Rank (mean-rank)"] == "minimize"

    assert np.all(df["Rank (mean-rank)"]["value"] == np.array([4, 3, 2, 1]))
