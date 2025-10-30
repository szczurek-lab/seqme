from typing import Literal

import numpy as np

from .exceptions import OptionalDependencyError


class AliphaticIndex:
    """Aliphatic index of peptide sequences.

    Installation: ``pip install "seqme[aa_descriptors]"``
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Computes the Aliphatic index.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            Aliphatic index for each sequence.
        """
        try:
            from modlamp.descriptors import GlobalDescriptor
        except ModuleNotFoundError:
            raise OptionalDependencyError("aa_descriptors") from None

        d = GlobalDescriptor(sequences)
        d.aliphatic_index()
        return d.descriptor.squeeze(axis=-1)


class Aromaticity:
    """Aromaticity of peptide sequences.

    Installation: ``pip install "seqme[aa_descriptors]"``
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Compute the aromaticity.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            Aromaticity value for each sequence.
        """
        try:
            from modlamp.descriptors import GlobalDescriptor
        except ModuleNotFoundError:
            raise OptionalDependencyError("aa_descriptors") from None

        d = GlobalDescriptor(sequences)
        d.aromaticity()
        return d.descriptor.squeeze(axis=-1)


class BomanIndex:
    """Boman index, estimating binding potential to proteins.

    Installation: ``pip install "seqme[aa_descriptors]"``
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Computes the Boman index.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            Boman index value for each sequence.
        """
        try:
            from modlamp.descriptors import GlobalDescriptor
        except ModuleNotFoundError:
            raise OptionalDependencyError("aa_descriptors") from None

        d = GlobalDescriptor(sequences)
        d.boman_index()
        return d.descriptor.squeeze(axis=-1)


class Charge:
    """Net charge of peptides at a given pH.

    Installation: ``pip install "seqme[aa_descriptors]"``
    """

    def __init__(self, ph: float = 7.0):
        """Initializes the Net charge.

        Args:
            ph: pH value at which to calculate the charge.
        """
        self.ph = ph

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Computes the net charge.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            Net charge for each sequence.
        """
        try:
            from modlamp.descriptors import GlobalDescriptor
        except ModuleNotFoundError:
            raise OptionalDependencyError("aa_descriptors") from None

        d = GlobalDescriptor(sequences)
        d.calculate_charge(ph=self.ph)
        return d.descriptor.squeeze(axis=-1)


class Gravy:
    """GRAVY (hydropathy) score for peptide sequences.

    Installation: ``pip install "seqme[aa_descriptors]"``
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Computes the GRAVY.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            GRAVY score for each sequence.
        """
        try:
            from modlamp.descriptors import PeptideDescriptor
        except ModuleNotFoundError:
            raise OptionalDependencyError("aa_descriptors") from None

        d = PeptideDescriptor(sequences)
        d.load_scale("gravy")
        d.calculate_global()
        return d.descriptor.squeeze(axis=-1)


class Hydrophobicity:
    """Hydrophobicity using a selected scale.

    Installation: ``pip install "seqme[aa_descriptors]"``
    """

    def __init__(self, scale: Literal["eisenberg", "hopp-woods", "janin", "kytedoolittle"] = "eisenberg"):
        """Initialize the hydrophobicity.

        Args:
            scale: Name of the hydrophobicity scale to use.
        """
        self.scale = scale

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Computes the hydrophobicity.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            Hydrophobicity score for each sequence.
        """
        try:
            from modlamp.descriptors import PeptideDescriptor
        except ModuleNotFoundError:
            raise OptionalDependencyError("aa_descriptors") from None

        d = PeptideDescriptor(sequences)
        d.load_scale(self.scale)
        d.calculate_global()
        return d.descriptor.squeeze(axis=-1)


class HydrophobicMoment:
    """Hydrophobic moment (i.e., amphiphilicity) for one or more peptide sequences using a sliding-window approach.

    Installation: ``pip install "seqme[aa_descriptors]"``
    """

    def __init__(
        self,
        scale: Literal["eisenberg", "hopp-woods", "janin", "kytedoolittle"] = "eisenberg",
        window: int = 11,
        angle: int = 100,
        modality: Literal["max", "mean"] = "mean",
    ):
        """Initialize the hydrophobic moment.

        Args:
            scale: Name of the hydrophobicity scale to use.
            window: Size of window
            angle: Angle in which to calculate the moment. 100 for alpha helices, 180 for beta sheets.
            modality: Method to compute statistic
        """
        self.scale = scale
        self.window = window
        self.angle = angle
        self.modality = modality

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Computes the hydrophobic moment.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            hydrophobic moment for each sequence.
        """
        try:
            from modlamp.descriptors import PeptideDescriptor
        except ModuleNotFoundError:
            raise OptionalDependencyError("aa_descriptors") from None

        d = PeptideDescriptor(sequences)
        d.load_scale(self.scale)
        d.calculate_moment(window=self.window, angle=self.angle, modality=self.modality)
        return d.descriptor.squeeze(axis=-1)


class InstabilityIndex:
    """Instability index, predicting in vitro protein stability.

    Installation: ``pip install "seqme[aa_descriptors]"``
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Computes the instability index.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            Instability index for each sequence.
        """
        try:
            from modlamp.descriptors import GlobalDescriptor
        except ModuleNotFoundError:
            raise OptionalDependencyError("aa_descriptors") from None

        d = GlobalDescriptor(sequences)
        d.instability_index()
        return d.descriptor.squeeze(axis=-1)


class IsoelectricPoint:
    """Isoelectric point of peptide sequences.

    Installation: ``pip install "seqme[aa_descriptors]"``
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Computes the isoelectric point of peptide sequences.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            Isoelectric point for each sequence.
        """
        try:
            from modlamp.descriptors import GlobalDescriptor
        except ModuleNotFoundError:
            raise OptionalDependencyError("aa_descriptors") from None

        d = GlobalDescriptor(sequences)
        d.isoelectric_point()
        return d.descriptor.squeeze(axis=-1)


class ProteinWeight:
    """Molecular weight of protein/peptide sequences.

    Installation: ``pip install "seqme[aa_descriptors]"``
    """

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Computes the molecular weight of peptide sequences.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            Molecular weight for each sequence.
        """
        try:
            from modlamp.descriptors import GlobalDescriptor
        except ModuleNotFoundError:
            raise OptionalDependencyError("aa_descriptors") from None

        d = GlobalDescriptor(sequences)
        d.calculate_MW()
        return d.descriptor.squeeze(axis=-1)
