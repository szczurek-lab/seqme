from typing import Literal

import numpy as np
from modlamp.descriptors import GlobalDescriptor, PeptideDescriptor


class AliphaticIndex:
    """
    Computes the aliphatic index of peptide sequences.
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Args:
            sequences: List of amino acid sequences.

        Returns:
            Aliphatic index for each sequence.
        """
        d = GlobalDescriptor(sequences)
        d.aliphatic_index()
        return d.descriptor.squeeze(axis=-1)


class Aromaticity:
    """
    Computes the aromaticity of peptide sequences.
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Args:
            sequences: List of amino acid sequences.

        Returns:
            Aromaticity value for each sequence.
        """
        d = GlobalDescriptor(sequences)
        d.aromaticity()
        return d.descriptor.squeeze(axis=-1)


class BomanIndex:
    """
    Computes the Boman index, estimating binding potential to proteins.
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Args:
            sequences: List of amino acid sequences.

        Returns:
            Boman index value for each sequence.
        """
        d = GlobalDescriptor(sequences)
        d.boman_index()
        return d.descriptor.squeeze(axis=-1)


class Charge:
    """
    Computes the net charge of peptides at a given pH.
    """

    def __init__(self, ph: float = 7.0):
        """
        Args:
            ph: pH value at which to calculate the charge.
        """
        self.ph = ph

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Args:
            sequences: List of amino acid sequences.

        Returns:
            Net charge for each sequence.
        """
        d = GlobalDescriptor(sequences)
        d.calculate_charge(ph=self.ph)
        return d.descriptor.squeeze(axis=-1)


class Gravy:
    """
    Computes the GRAVY (hydropathy) score for peptide sequences.
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Args:
            sequences: List of amino acid sequences.

        Returns:
            GRAVY score for each sequence.
        """
        d = PeptideDescriptor(sequences)
        d.load_scale("gravy")
        d.calculate_global()
        return d.descriptor.squeeze(axis=-1)


class Hydrophobicity:
    """
    Computes hydrophobicity using a selected scale.
    """

    def __init__(self, scale: Literal["eisenberg", "hopp-woods", "janin", "kytedoolittle"] = "eisenberg"):
        """
        Args:
            scale: Name of the hydrophobicity scale to use.
        """
        self.scale = scale

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Args:
            sequences: List of amino acid sequences.

        Returns:
            Hydrophobicity score for each sequence.
        """
        d = PeptideDescriptor(sequences)
        d.load_scale(self.scale)
        d.calculate_global()
        return d.descriptor.squeeze(axis=-1)


class InstabilityIndex:
    """
    Computes the instability index, predicting in vitro protein stability.
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Args:
            sequences: List of amino acid sequences.

        Returns:
            Instability index for each sequence.
        """
        d = GlobalDescriptor(sequences)
        d.instability_index()
        return d.descriptor.squeeze(axis=-1)


class IsoelectricPoint:
    """
    Computes the isoelectric point of peptide sequences.
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Args:
            sequences: List of amino acid sequences.

        Returns:
            Isoelectric point for each sequence.
        """
        d = GlobalDescriptor(sequences)
        d.isoelectric_point()
        return d.descriptor.squeeze(axis=-1)


class MolecularWeight:
    """
    Computes the molecular weight of peptide sequences.
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Args:
            sequences: List of amino acid sequences.

        Returns:
            Molecular weight for each sequence.
        """
        d = GlobalDescriptor(sequences)
        d.calculate_MW()
        return d.descriptor.squeeze(axis=-1)
