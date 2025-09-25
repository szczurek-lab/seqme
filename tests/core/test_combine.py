import pytest

from seqme import combine, score
from seqme.metrics import Novelty, Uniqueness


def test_mean_on_overlap():
    metrics = [Uniqueness()]

    sequences1 = {"my_model": ["KKW", "RRR"]}
    df1 = score(sequences1, metrics)

    sequences2 = {"my_model": ["KKW", "KKW"]}
    df2 = score(sequences2, metrics)

    df = combine([df1, df2], on_overlap="mean,std")
    assert df.loc[("my_model", ("Uniqueness", "value"))] == 0.75
    assert df.loc[("my_model", ("Uniqueness", "deviation"))] == 0.25

    assert df.shape == (1, 2)


def test_fail_on_overlap():
    metrics = [Novelty(reference=["KKW"])]
    sequences = {"my_model": ["KKW", "RRR", "RRR"]}

    df1 = score(sequences, metrics)
    df2 = score(sequences, metrics)

    with pytest.raises(ValueError) as e:
        combine([df1, df2], on_overlap="fail")

    assert str(e.value) == "Multiple values in cell: [my_model, ('Novelty', 'value')]"
