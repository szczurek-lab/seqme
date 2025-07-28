import pandas as pd
import pytest

from seqme import compute_metrics
from seqme.metrics import Novelty


def test_compute_metrics():
    sequences = {"my_model": ["KKW", "RRR", "RRR"]}
    metrics = [Novelty(reference=["KKW"])]

    df = compute_metrics(sequences, metrics)

    # shape
    assert df.shape == (1, 2)
    # objective metadata
    assert df.attrs["objective"] == {"Novelty": "maximize"}
    # index & columns
    assert df.index.tolist() == ["my_model"]
    assert df.columns.tolist() == [("Novelty", "value"), ("Novelty", "deviation")]
    # value & deviation
    assert df.at["my_model", ("Novelty", "value")] == pytest.approx(2.0 / 3.0)
    assert pd.isna(df.at["my_model", ("Novelty", "deviation")])


def test_empty_metrics_list_raises():
    sequences = {"random": ["MKQW", "RKSPL"]}
    metrics = []

    with pytest.raises(ValueError, match=r"^No metrics provided$"):
        compute_metrics(sequences, metrics)


def test_duplicate_metric_names_raises():
    sequences = {"random": ["MKQW", "RKSPL"]}

    metrics = [
        Novelty(reference=["KKW", "RKSPL"]),
        Novelty(reference=["RASD", "KKKQ", "LPTUY"]),
    ]

    with pytest.raises(
        ValueError,
        match=r"^Metrics must have unique names\. Found duplicates: Novelty, Novelty$",
    ):
        compute_metrics(sequences, metrics)
