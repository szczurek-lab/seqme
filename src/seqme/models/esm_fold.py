from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from .exceptions import OptionalDependencyError


class EsmFold:
    """
    ESMFold protein language model.

    The model predicts the 3D-structure (fold) of a protein sequence.

    Installation: ``pip install seqme[esm2]``

    Reference:
        Lin et al., "Language models of protein sequences at the scale of evolution enable accurate structure prediction"
        (https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int = 256,
        verbose: bool = False,
    ):
        """
        Initialize the ESMFold model.

        Args:
            device: Device to run inference on, e.g., "cuda" or "cpu".
            batch_size: Number of sequences to process per batch.
            verbose: Whether to display a progress bar.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        try:
            from transformers import AutoTokenizer, EsmForProteinFolding
        except ModuleNotFoundError:
            raise OptionalDependencyError("esm2") from None

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")

        self.model.to(device)
        self.model.eval()

    def __call__(self, sequences: list[str]) -> list[np.ndarray]:
        return self.fold(sequences)

    def fold(self, sequences: list[str], convention: Literal["atom14", "ca"] = "ca") -> list[np.ndarray]:
        """
        Predict protein sequences 3D-structure, i.e., atom coordinates.

        The atom position / coordinate encoding corresponds is 'atom14':

            residue_atoms: dict[str, list[str]] = {
                "ALA": ["C", "CA", "CB", "N", "O"],
                "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
                "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
                "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
                "CYS": ["C", "CA", "CB", "N", "O", "SG"],
                "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
                "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
                "GLY": ["C", "CA", "N", "O"],
                "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
                "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
                "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
                "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
                "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
                "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
                "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
                "SER": ["C", "CA", "CB", "N", "O", "OG"],
                "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
                "TRP": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "N", "NE1", "O"],
                "TYR": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH"],
                "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
            }

            The atom14 mapping is from here: https://github.com/huggingface/transformers/blob/99b0995138c17ef953959c70f35cb2bdc41111a2/src/transformers/models/esm/openfold_utils/residue_constants.py#L335

            Amino-acids are defined by at most 14 atoms (excluding hydrogens). The shape of a single sequence fold is [sequence_length, 14, 3],
            where the dimension with 14 elements, corresponds to atom positions of an amino acid, and the dimension with 3 elements corresponds to xyz. If an amino-acid has fewer than 14 atoms, then those positions should be discarded / ignored.

        Args:
            sequences: List of input amino acid sequences.
            convention: Whether to return atom14 or the carbon alphas position of each amino acid ("ca").

        Returns:
            A list of variable length numpy array of shape (sequence_length, convention (1 or 14), 3 (xyz))
        """
        folds = []
        with torch.inference_mode():
            for i in tqdm(
                range(0, len(sequences), self.batch_size),
                disable=not self.verbose,
            ):
                batch = sequences[i : i + self.batch_size]
                tokens = self.tokenizer(
                    batch, return_tensors="pt", add_special_tokens=False, padding=True, truncation=False
                )
                tokens = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}

                outputs = self.model(**tokens)

                # @TODO: Return the atom14 mask as well and return a dict.
                # print(outputs["atom14_atom_exists"])

                lengths = [len(s) for s in batch]
                if convention == "ca":
                    folded_positions = [outputs.positions[-1, i, :length, 1, :] for i, length in enumerate(lengths)]
                elif convention == "atom14":
                    folded_positions = [outputs.positions[-1, i, :length, :, :] for i, length in enumerate(lengths)]
                else:
                    raise ValueError(f"Unsupported convention: '{convention}'.")

                folds += folded_positions

        return folds
