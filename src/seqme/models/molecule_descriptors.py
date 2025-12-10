import os

import numpy as np

from .exceptions import OptionalDependencyError


class SAScore:
    """Synthetic Accessibility Score (SA Score) for SMILES sequences.

    Installation: ``pip install "seqme[molecule_descriptors]"``

    Reference:
        P. Ertl and A. Schuffenhauer, "Estimation of synthetic accessibility score of drug-like molecules based on molecular complexity and fragment contributions"
        (http://www.jcheminf.com/content/1/1/8)
    """

    def __init__(self):
        try:
            import importlib.util

            from rdkit.Chem import RDConfig
        except ModuleNotFoundError:
            raise OptionalDependencyError("molecule_descriptors") from None

        path = os.path.join(RDConfig.RDContribDir, "SA_Score", "sascorer.py")
        assert os.path.isfile(path), "sascorer.py does not exist."

        spec = importlib.util.spec_from_file_location("sascorer", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._sascorer = module
        assert hasattr(self._sascorer, "calculateScore"), "sascorer module has no expected function."

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Compute SA score for a list of SMILES sequences.

        Args:
            sequences: List of SMILES strings.

        Returns:
            SA-score for each sequence.
        """
        try:
            from rdkit import Chem
        except ModuleNotFoundError:
            raise OptionalDependencyError("molecule_descriptors") from None

        return np.array([self._sascorer.calculateScore(Chem.MolFromSmiles(sequence)) for sequence in sequences])


class QED:
    """Quantitative Estimate of Drug-likeness for SMILES sequences.

    Installation: ``pip install "seqme[molecule_descriptors]"``

    Reference:
        G. Richard Bickerton et al., "Quantifying the chemical beauty of drugs"
        (https://www.nature.com/articles/nchem.1243)
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Compute QED of SMILES sequence.

        Args:
            sequences: SMILES sequences.

        Returns:
            QED for each sequence.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem.QED import qed
        except ModuleNotFoundError:
            raise OptionalDependencyError("molecule_descriptors") from None

        return np.array([qed(Chem.MolFromSmiles(sequence)) for sequence in sequences])


class LogP:
    """Lipophilicity for SMILES sequences.

    Installation: ``pip install "seqme[molecule_descriptors]"``

    Reference:
        S. A. Wildman and Gordon M. Crippen, "Prediction of Physicochemical Parameters by Atomic Contributions"
        (https://pubs.acs.org/doi/10.1021/ci990307l)
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Compute lipophilicity of SMILES sequence.

        Args:
            sequences: SMILES sequences.

        Returns:
            Log-P for each sequence.
        """
        try:
            from rdkit import Chem
            from rdkit.Chem.Descriptors import MolLogP  # type: ignore
        except ModuleNotFoundError:
            raise OptionalDependencyError("molecule_descriptors") from None

        return np.array([MolLogP(Chem.MolFromSmiles(sequence)) for sequence in sequences])
