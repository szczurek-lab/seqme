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
    df = sm.extract_non_dominated(df)

    assert df.index.to_list() == ["model3", "model4"]


def test_nested():
    sequences = {
        ("model1", 1): ["A"],
        ("model1", 2): ["AA", "BB", "CC"],
        ("model2", 1): ["AAA", "BBB", "CCC"],
        ("model2", 2): ["AA", "BB", "CC", "DDD"],
    }
    metrics = [
        sm.metrics.Count(),
        sm.metrics.Length(objective="maximize"),
    ]

    df = sm.evaluate(sequences, metrics)
    df = sm.extract_non_dominated(df, level=1)

    assert df.index.tolist() == [("model1", 2), ("model2", 1), ("model2", 2)]
