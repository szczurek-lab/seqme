from enum import Enum
from itertools import islice

import numpy as np
import torch
from tqdm import tqdm

from .exceptions import OptionalDependencyError


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
    Wrapper for the ESM2 protein/peptide embedding model from HuggingFace.

    Computes sequence-level embeddings by averaging token embeddings,
    excluding [CLS] and [EOS] tokens.

    Installation: ``pip install seqme[esm2]``

    Reference:
        Lin et al., "Language models of protein sequences at the scale of evolution enable accurate structure prediction"
        (https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)
    """

    def __init__(
        self,
        model_name: Esm2Checkpoint | str,
        *,
        device: str | None = None,
        batch_size: int = 256,
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

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
            from transformers.utils import logging
        except ModuleNotFoundError:
            raise OptionalDependencyError("esm2") from None

        prev = logging.get_verbosity()
        logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        logging.set_verbosity(prev)

        self.model.to(device)
        self.model.eval()

    def __call__(self, sequences: list[str]) -> np.ndarray:
        return self.embed(sequences)

    def embed(self, sequences: list[str], layer: int = -1) -> np.ndarray:
        """
        Compute embeddings for a list of sequences.

        Each sequence is tokenized and passed through the model.
        Token embeddings are averaged (excluding special tokens) to produce a single embedding per sequence.

        Args:
            sequences: List of input amino acid sequences.
            layer: Layer to retrieve embeddings from.

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
                hidden_state = self.model(**tokens, output_hidden_states=True).hidden_states[layer]

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

    def compute_pseudo_perplexity(self, sequences: list[str], mask_size: int = 1) -> np.ndarray:
        """
        Compute pseudo-perplexity for a list of sequences, masking `mask_size` positions per pass.

        Args:
            sequences: List of amino acid sequences.
            mask_size: Number of tokens to mask simultaneously in each forward pass.

        Returns:
            np.ndarray: Pseudo-perplexity scores, in the same order as the input sequences.
        """
        inputs = self.tokenizer(sequences, return_tensors="pt", padding=True, truncation=False)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        B, L = input_ids.size()

        total_loglik = torch.zeros(B, device=self.device)
        lengths = attention_mask.sum(dim=1)

        valid_positions = [pos for pos in range(L) if attention_mask[:, pos].any()]

        # Utility to chunk a list into size‐<=mask_size
        def chunked(lst, n):
            it = iter(lst)
            while True:
                chunk = list(islice(it, n))
                if not chunk:
                    break
                yield chunk

        for pos_chunk in chunked(valid_positions, mask_size):
            masked_in = input_ids.clone()

            for pos in pos_chunk:
                real = attention_mask[:, pos] == 1
                masked_in[real, pos] = self.tokenizer.mask_token_id

            with torch.no_grad():
                logits = self.model(masked_in, attention_mask=attention_mask.to(self.device)).logits
                log_probs = torch.log_softmax(logits, dim=-1)

            for pos in pos_chunk:
                real = attention_mask[:, pos] == 1
                true_ids = input_ids[:, pos]
                pos_logps = log_probs[torch.arange(B, device=self.device), pos, true_ids]
                total_loglik[real] += pos_logps[real]

        pppls = torch.exp(-total_loglik / lengths).cpu().numpy()
        return pppls
