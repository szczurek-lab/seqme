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


def test_mean_rank_tie():
    sequences = {
        "model1": ["AA", "BB", "CC"],
        "model2": ["A"],
        "model3": ["AA", "BB", "CC", "DDD"],
        "model4": ["AA", "BB", "CC", "DDD"],
        "model5": ["AA", "BB", "CC"],
    }
    metrics = [
        sm.metrics.Count(),
        sm.metrics.Length(objective="maximize"),
    ]

    df = sm.evaluate(sequences, metrics)

    df2 = sm.rank(df, tiebreak="mean-rank", name="Rank (mean-rank)", ties="min")
    assert np.all(df2["Rank (mean-rank)"]["value"] == np.array([3, 5, 1, 1, 3]))

    df2 = sm.rank(df, tiebreak="mean-rank", name="Rank (mean-rank)", ties="max")
    assert np.all(df2["Rank (mean-rank)"]["value"] == np.array([4, 5, 2, 2, 4]))

    df2 = sm.rank(df, tiebreak="mean-rank", name="Rank (mean-rank)", ties="mean")
    assert np.all(df2["Rank (mean-rank)"]["value"] == np.array([3.5, 5, 1.5, 1.5, 3.5]))

    df2 = sm.rank(df, tiebreak="mean-rank", name="Rank (mean-rank)", ties="dense")
    assert np.all(df2["Rank (mean-rank)"]["value"] == np.array([2, 3, 1, 1, 2]))

    df2 = sm.rank(df, tiebreak="mean-rank", name="Rank (mean-rank)", ties="auto")
    assert np.all(df2["Rank (mean-rank)"]["value"] == np.array([3, 5, 1, 1, 3]))


def test_single_tie():
    sequences = {
        "model1": ["AA", "BB", "CC"],
    }
    metrics = [
        sm.metrics.Count(),
        sm.metrics.Length(objective="maximize"),
    ]

    df = sm.evaluate(sequences, metrics)

    df2 = sm.rank(df, tiebreak="mean-rank", name="Rank (mean-rank)", ties="min")
    assert np.all(df2["Rank (mean-rank)"]["value"] == np.array([1]))

    df2 = sm.rank(df, tiebreak="mean-rank", name="Rank (mean-rank)", ties="max")
    assert np.all(df2["Rank (mean-rank)"]["value"] == np.array([1]))

    df2 = sm.rank(df, tiebreak="mean-rank", name="Rank (mean-rank)", ties="mean")
    assert np.all(df2["Rank (mean-rank)"]["value"] == np.array([1]))

    df2 = sm.rank(df, tiebreak="mean-rank", name="Rank (mean-rank)", ties="dense")
    assert np.all(df2["Rank (mean-rank)"]["value"] == np.array([1]))

    df2 = sm.rank(df, tiebreak="mean-rank", name="Rank (mean-rank)", ties="auto")
    assert np.all(df2["Rank (mean-rank)"]["value"] == np.array([1]))
