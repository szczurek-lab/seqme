from seqme.metrics import Count


def test_count_properties():
    metric = Count()
    assert metric.name == "Count"
    assert metric.objective == "maximize"


def test_count_value():
    metric = Count()
    sequences = ["AAA", "BBBB", "CCCCC"]
    result = metric(sequences)
    assert result.value == 3
