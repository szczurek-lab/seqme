import seqme as sm


def test_rename_single():
    sequences = {
        "model1": ["AAA", "AAA", "AAA"],
        "model2": ["AAA", "KKK", "KKK"],
    }
    metrics = [sm.metrics.Uniqueness()]

    df = sm.evaluate(sequences, metrics)

    df2 = sm.rename(df, {"Uniqueness": "UQ"})

    assert df.shape == (2, 2)
    assert df.attrs["objective"] == {"Uniqueness": "maximize"}
    assert df.index.tolist() == ["model1", "model2"]
    assert df.columns.tolist() == [("Uniqueness", "value"), ("Uniqueness", "deviation")]

    assert df2.shape == (2, 2)
    assert df2.attrs["objective"] == {"UQ": "maximize"}
    assert df2.index.tolist() == ["model1", "model2"]
    assert df2.columns.tolist() == [("UQ", "value"), ("UQ", "deviation")]


def test_rename_swap():
    sequences = {
        "model1": ["AAA", "AAA", "AAA"],
        "model2": ["AAA", "KKK", "KKK"],
    }
    metrics = [
        sm.metrics.Uniqueness(),
        sm.metrics.NGramJaccardSimilarity(reference=["A"], n=1, objective="minimize"),
    ]

    df = sm.evaluate(sequences, metrics)

    df2 = sm.rename(df, {"Uniqueness": "Jaccard-similarity", "Jaccard-similarity": "Uniqueness"})

    assert df.shape == (2, 4)
    assert df.attrs["objective"] == {"Uniqueness": "maximize", "Jaccard-similarity": "minimize"}
    assert df.index.tolist() == ["model1", "model2"]
    assert df.columns.tolist() == [
        ("Uniqueness", "value"),
        ("Uniqueness", "deviation"),
        ("Jaccard-similarity", "value"),
        ("Jaccard-similarity", "deviation"),
    ]

    assert df2.shape == (2, 4)
    assert df2.attrs["objective"] == {"Jaccard-similarity": "maximize", "Uniqueness": "minimize"}
    assert df2.index.tolist() == ["model1", "model2"]
    assert df2.columns.tolist() == [
        ("Jaccard-similarity", "value"),
        ("Jaccard-similarity", "deviation"),
        ("Uniqueness", "value"),
        ("Uniqueness", "deviation"),
    ]
