from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from .exceptions import OptionalDependencyError


class EsmFold:
    """
    Wrapper for the ESMFold protein model from HuggingFace.

    Predicts a protein sequences 3D-structure.

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
        Initialize the ESM2 model.

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
        Predict sequences 3D-structure, i.e., atom coordinates.

        Args:
            sequences: List of input amino acid sequences.
            convention: Whether to return atom14 or c_alpha.

        Returns:
            A list of variable length numpy array of length (sequence_length, convention)
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

                # @TODO: atom14 mask extract the correct ones.
                # print(outputs["atom14_atom_exists"])

                # https://github.com/huggingface/transformers/blob/99b0995138c17ef953959c70f35cb2bdc41111a2/src/transformers/models/esm/openfold_utils/residue_constants.py#L335

                lengths = [len(s) for s in batch]

                if convention == "ca":
                    folded_positions = [outputs.positions[-1, i, :length, 1, :] for i, length in enumerate(lengths)]
                elif convention == "atom14":
                    folded_positions = [outputs.positions[-1, i, :length, :, :] for i, length in enumerate(lengths)]
                else:
                    raise ValueError(f"Unsupported convention: '{convention}'.")

                folds += folded_positions

        return folds
