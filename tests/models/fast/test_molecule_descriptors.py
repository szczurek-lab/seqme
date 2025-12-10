import numpy as np
import pytest

from seqme.models import QED, LogP, SAScore

pytest.importorskip("rdkit")


def test_descriptors():
    smiles = ["CCO", "c1ccccc1"]

    ms = [LogP(), SAScore(), QED()]
    vs = np.array([m(smiles) for m in ms])

    assert vs.shape == (3, 2)
