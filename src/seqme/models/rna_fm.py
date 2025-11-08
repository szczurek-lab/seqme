from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from .exceptions import OptionalDependencyError


class RNAFM:
    """
    A language model trained on RNA sequences, which computes sequence-level embeddings by averaging token embeddings.

    Two checkpoints are available:
        mRNA: 239M parameters, 12 layers, embedding dim 1280, trained on 45 million mRNA coding sequences (CDS). Must be codon aligned.
        ncRNA: 99M parameters, 12 layers, embedding dim 640, trained on 23.7 million non-coding RNA (ncRNA) sequences.

    Installation: ``pip install "seqme[rnafm]"``

    Reference:
        Chen et al., "Interpretable RNA Foundation Model from Unannotated Data for Highly Accurate RNA Structure and Function Predictions"
        (https://arxiv.org/pdf/2204.00300)

    """

    def __init__(
        self,
        *,
        model_name: Literal["mRNA", "ncRNA"] = "mRNA",
        device: str | None = None,
        batch_size: int = 256,
        verbose: bool = False,
    ):
        """
        Initialize model.

        Args:
            model_name: Either a mRNA or ncRNA checkpoint.
            device: Device to run inference on, e.g., ``"cuda"`` or ``"cpu"``.
            batch_size: Number of sequences to process per batch.
            verbose: Whether to display a progress bar.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        try:
            import fm
        except ModuleNotFoundError:
            raise OptionalDependencyError("rnafm") from None

        model, alphabet = fm.pretrained.rna_fm_t12() if self.model_name == "ncRNA" else fm.pretrained.mrna_fm_t12()
        batch_converter = alphabet.get_batch_converter()

        self.model = model.to(device).eval()
        self.batch_converter = batch_converter

    def __call__(self, sequences: list[str]) -> np.ndarray:
        return self.embed(sequences)

    @torch.inference_mode()
    def embed(self, sequences: list[str], layer: int = 12) -> np.ndarray:
        """
        Compute embeddings for the RNA sequences.

        Each sequence is tokenized and passed through the model.
        Token embeddings are averaged to produce a single embedding per sequence.

        Args:
            sequences: RNA sequences to embed.
            layer: Embedding layer. Last layer is 12.

        Returns:
            A NumPy array of shape (n_sequences, embedding_dim) containing the embeddings.
        """
        if self.model_name == "mRNA":
            for sequence in sequences:
                if len(sequence) % 3 != 0:
                    raise ValueError(f"Found non-codon aligned sequence with {len(sequence)}) nucleotides.")

        embeddings = []
        for i in tqdm(range(0, len(sequences), self.batch_size), disable=not self.verbose):
            batch = sequences[i : i + self.batch_size]

            named_batch = [("", b) for b in batch]
            tokens = self.batch_converter(named_batch)[2].to(self.device)

            results = self.model(tokens, repr_layers=[layer])
            hidden_state = results["representations"][layer]

            lengths = [len(s) // 3 if self.model_name == "mRNA" else len(s) for s in batch]
            means = [hidden_state[i, :length].mean(dim=-2) for i, length in enumerate(lengths)]
            embed = torch.stack(means, dim=0)

            embeddings.append(embed.cpu().numpy())

        return np.concatenate(embeddings)
