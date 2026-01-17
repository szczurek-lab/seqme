import numpy as np

import seqme as sm


def test_drop_deviation():
    sequences = {
        "model1": ["A"],
        "model2": ["AA", "BB", "CC"],
        "model3": ["AAA", "BBB", "CCC"],
        "model4": ["AA", "BB", "CC", "DDD"],
    }
    metrics = [
        sm.metrics.Length(objective="maximize"),
    ]

    df = sm.evaluate(sequences, metrics)
    assert df.shape == (4, 1 * 2)

    df = sm.drop(df, ["Length"])

    deviation_column = ("Length", "deviation")
    assert np.isnan(df[deviation_column].values).all()


def test_drop_both():
    sequences = {
        "model1": ["A"],
        "model2": ["AA", "BB", "CC"],
        "model3": ["AAA", "BBB", "CCC"],
        "model4": ["AA", "BB", "CC", "DDD"],
    }
    metrics = [
        sm.metrics.Length(objective="maximize"),
    ]

    df = sm.evaluate(sequences, metrics)
    assert df.shape == (4, 1 * 2)

    df = sm.drop(df, ["Length"], "both")

    assert "Length" not in df.columns.get_level_values(0).unique()
