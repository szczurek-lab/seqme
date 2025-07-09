import numpy as np
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from peptides import Peptide


class Gravy:
    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([ProteinAnalysis(seq).gravy() for seq in sequences])


class Charge:
    def __init__(self, ph: float = 7.0):
        self.ph = ph

    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([ProteinAnalysis(seq).charge_at_pH(self.ph) for seq in sequences])


class Hydrophobicity:
    def __init__(self, scale: str = "Aboderin"):
        self.scale = scale

    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([Peptide(seq).hydrophobicity(scale=self.scale) for seq in sequences])


class IsoelectricPoint:
    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([ProteinAnalysis(seq).isoelectric_point() for seq in sequences])


class MolecularWeight:
    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([ProteinAnalysis(seq).molecular_weight() for seq in sequences])


class InstabilityIndex:
    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([ProteinAnalysis(seq).instability_index() for seq in sequences])


class AliphaticIndex:
    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([Peptide(seq).aliphatic_index() for seq in sequences])


class BomanIndex:
    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([Peptide(seq).boman() for seq in sequences])
