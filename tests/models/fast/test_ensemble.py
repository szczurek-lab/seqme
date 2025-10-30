import numpy as np

import seqme as sm


def biased_model(model, offset: float):
    return lambda seqs: model(seqs) + offset


def test_ensemble_equal():
    sequences = ["KKK", "KKKK", "KKKKK"]

    ensemble = sm.models.Ensemble([sm.models.Charge(), biased_model(sm.models.Charge(), offset=1)])
    values = ensemble(sequences)

    assert values.shape == (3,)
    assert np.array_equal(values, [3.495, 4.495, 5.495])


def test_ensemble_weighted():
    sequences = ["KKK", "KKKK", "KKKKK"]

    ensemble = sm.models.Ensemble([sm.models.Charge(), biased_model(sm.models.Charge(), offset=1)], weights=[1, 2])
    values = ensemble(sequences)

    assert values.shape == (3,)
    assert np.array_equal(values, [10.985, 13.985, 16.985])
