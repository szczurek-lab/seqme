import seqme as sm


def test_sort_best():
    sequences = {
        "model1": ["AAA", "KKK", "AAA"],
        "model2": ["AAA", "KKK", "RRR"],
        "model3": ["AAA", "AAA", "AAA"],
    }
    metrics = [sm.metrics.Uniqueness()]

    df = sm.evaluate(sequences, metrics)
    df = sm.sort(df, metric="Uniqueness", order="best")

    assert df.shape == (3, 2)
    assert df.attrs["objective"] == {"Uniqueness": "maximize"}

    assert df.index.tolist() == ["model2", "model1", "model3"]
    assert df.columns.tolist() == [("Uniqueness", "value"), ("Uniqueness", "deviation")]


def test_sort_worst():
    sequences = {
        "model1": ["AAA", "KKK", "AAA"],
        "model2": ["AAA", "KKK", "RRR"],
        "model3": ["AAA", "AAA", "AAA"],
    }
    metrics = [sm.metrics.Uniqueness()]

    df = sm.evaluate(sequences, metrics)
    df = sm.sort(df, metric="Uniqueness", order="worst")

    assert df.shape == (3, 2)
    assert df.attrs["objective"] == {"Uniqueness": "maximize"}

    assert df.index.tolist() == ["model3", "model1", "model2"]
    assert df.columns.tolist() == [("Uniqueness", "value"), ("Uniqueness", "deviation")]
