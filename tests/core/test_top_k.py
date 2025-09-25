import pandas as pd
import pytest

import seqme as sm


def test_top_k():
    sequences = {
        "model1": ["AAA", "AAA", "AAA"],
        "model2": ["AAA", "KKK", "KKK"],
        "model3": ["AAA", "KKK", "RRR"],
    }
    metrics = [sm.metrics.Uniqueness()]

    df = sm.evaluate(sequences, metrics)
    df = sm.top_k(df, metric="Uniqueness", k=2)

    assert df.shape == (2, 2)
    assert df.attrs["objective"] == {"Uniqueness": "maximize"}

    assert df.index.tolist() == ["model2", "model3"]
    assert df.columns.tolist() == [("Uniqueness", "value"), ("Uniqueness", "deviation")]

    assert df.at["model2", ("Uniqueness", "value")] == pytest.approx(2 / 3)
    assert df.at["model3", ("Uniqueness", "value")] == 1

    assert pd.isna(df.at["model2", ("Uniqueness", "deviation")])
    assert pd.isna(df.at["model3", ("Uniqueness", "deviation")])
