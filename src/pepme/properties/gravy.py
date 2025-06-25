import numpy as np
import peptides


class Gravy:
    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([peptides.Peptide(seq).hydrophobicity() for seq in sequences])
