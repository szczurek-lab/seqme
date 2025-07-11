from enum import Enum

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging


class Esm2Checkpoint(str, Enum):
    # protein checkpoints
    t6_8M = "facebook/esm2_t6_8M_UR50D"
    t12_35M = "facebook/esm2_t12_35M_UR50D"
    t30_150M = "facebook/esm2_t30_150M_UR50D"
    t33_650M = "facebook/esm2_t33_650M_UR50D"
    t36_3B = "facebook/esm2_t36_3B_UR50D"
    t48_15B = "facebook/esm2_t48_15B_UR50D"

    # peptide checkpoints
    shukla_group_peptide_650M = "ShuklaGroupIllinois/PeptideESM2_650M"


class Esm2:
    """
    Wrapper for the ESM2 protein/peptide embedding model from Hugging Face.

    Computes sequence-level embeddings by averaging token embeddings,
    excluding [CLS] and [EOS] tokens.
    """

    def __init__(
        self,
        model_name: Esm2Checkpoint | str,
        device: str,
        batch_size: int,
        verbose: bool = False,
    ):
        """
        Initialize the ESM2 model.

        Args:
            model_name: Model checkpoint name or enum.
            device: Device to run inference on, e.g., "cuda" or "cpu".
            batch_size: Number of sequences to process per batch.
            verbose: Whether to display a progress bar.
        """
        if isinstance(model_name, Esm2Checkpoint):
            model_name = model_name.value

        prev = logging.get_verbosity()
        logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        logging.set_verbosity(prev)

        self.model.to(device)

        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Compute embeddings for a list of sequences.

        Each sequence is tokenized and passed through the model.
        Token embeddings are averaged (excluding special tokens) to produce a single embedding per sequence.

        Args:
            sequences: List of input amino acid sequences.

        Returns:
            A NumPy array of shape (n_sequences, embedding_dim) containing the embeddings.
        """
        embeddings = []
        with torch.inference_mode():
            for i in tqdm(
                range(0, len(sequences), self.batch_size),
                disable=not self.verbose,
            ):
                batch = sequences[i : i + self.batch_size]
                tokens = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
                tokens = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}
                hidden_state = self.model(**tokens).last_hidden_state

                counts = tokens["attention_mask"].sum(dim=-1)
                mask = tokens["attention_mask"]

                batch_size = mask.size(0)
                batch_indices = torch.arange(batch_size, device=mask.device)
                mask[batch_indices, 0] = 0
                mask[batch_indices, counts - 1] = 0
                counts = counts - 2

                embed = (hidden_state * mask.unsqueeze(-1)).sum(dim=-2) / counts.unsqueeze(-1)
                embeddings.append(embed.cpu().numpy())

        return np.concatenate(embeddings)
