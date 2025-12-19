import pytest

from seqme import combine, evaluate
from seqme.metrics import Novelty, Uniqueness


def test_mean_on_overlap():
    metrics = [Uniqueness()]

    sequences1 = {"my_model": ["KKW", "RRR"]}
    df1 = evaluate(sequences1, metrics)

    sequences2 = {"my_model": ["KKW", "KKW"]}
    df2 = evaluate(sequences2, metrics)

    df = combine([df1, df2], value="mean", deviation="std")
    assert df.loc[("my_model", ("Uniqueness", "value"))] == 0.75
    assert df.loc[("my_model", ("Uniqueness", "deviation"))] == pytest.approx(0.35355339)

    assert df.shape == (1, 2)


def test_fail_on_overlap():
    metrics = [Novelty(reference=["KKW"])]
    sequences = {"my_model": ["KKW", "RRR", "RRR"]}

    df1 = evaluate(sequences, metrics)
    df2 = evaluate(sequences, metrics)

    with pytest.raises(ValueError) as e:
        combine([df1, df2], value=None)

    assert str(e.value) == "Multiple values in cell: [my_model, Novelty]."
