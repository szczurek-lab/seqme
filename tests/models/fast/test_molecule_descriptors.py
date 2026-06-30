import numpy as np
import pytest

from seqme.metrics import ID
from seqme.models import QED, LogP, MoleculeValidity, SAScore

pytest.importorskip("rdkit")


def test_descriptors():
    smiles = ["CCO", "c1ccccc1"]

    ms = [LogP(), SAScore(), QED()]
    vs = np.array([m(smiles) for m in ms])

    assert vs.shape == (3, 2)


def test_molecule_validity():
    smiles = ["CCO", "c1ccccc1", "not_a_smiles"]

    validity = MoleculeValidity()(smiles)

    assert validity.dtype == np.bool_
    np.testing.assert_array_equal(validity, [True, True, False])


def test_molecule_validity_with_id_metric():
    smiles = ["CCO", "c1ccccc1", "not_a_smiles"]

    metric = ID(predictor=MoleculeValidity(), name="Validity", objective="maximize")
    result = metric(smiles)

    assert result.value == pytest.approx(2 / 3)
