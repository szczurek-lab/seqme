from enum import Enum
from itertools import islice

import numpy as np
import torch
from tqdm import tqdm

from .exceptions import OptionalDependencyError


class ESM2Checkpoint(str, Enum):
    """
    ESM-2 checkpoints.

    Available checkpoints:
        t6_8M: 8M parameters, 6 layers, embedding dim 320 - compact variant for quick prototyping and resource-constrained inference.
        t12_35M: 35M parameters, 12 layers, embedding dim 480 - mid-size variant balancing compute and performance.
        t30_150M: 150M parameters, 30 layers, embedding dim 640 - larger variant that improves representation power for downstream tasks.
        t33_650M: 650M parameters, 33 layers, embedding dim 1280, commonly used medium-large model that performs well on structure and property prediction tasks.
        t36_3B: 3B parameters, 36 layers, embedding dim 2560 - large model for more accurate representations and structure inference.
        t48_15B: 15B parameters, 48 layers, embedding dim 5120 - the largest public ESM-2 variant; offers the highest capacity and best single-sequence structure representational power.

        shukla_group_peptide_650M: 650M parameters, 33 layers, embedding dim 1280, trained on peptide sequences.
    """

    # protein checkpoints
    t6_8M = "facebook/esm2_t6_8M_UR50D"
    t12_35M = "facebook/esm2_t12_35M_UR50D"
    t30_150M = "facebook/esm2_t30_150M_UR50D"
    t33_650M = "facebook/esm2_t33_650M_UR50D"
    t36_3B = "facebook/esm2_t36_3B_UR50D"
    t48_15B = "facebook/esm2_t48_15B_UR50D"

    # peptide checkpoints
    shukla_group_peptide_650M = "ShuklaGroupIllinois/PeptideESM2_650M"


class ESM2:
    """
    Wrapper for the ESM2 protein/peptide embedding model.

    Computes sequence-level embeddings by averaging token embeddings excluding [CLS] and [EOS] tokens.

    Installation: ``pip install "seqme[esm2]"``

    Reference:
        Lin et al., "Language models of protein sequences at the scale of evolution enable accurate structure prediction"
        (https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)
    """

    def __init__(
        self,
        model_name: ESM2Checkpoint | str,
        *,
        device: str | None = None,
        batch_size: int = 256,
        cache_dir: str | None = None,
        verbose: bool = False,
    ):
        """
        Initialize the model.

        Args:
            model_name: Model checkpoint name or enum.
            device: Device to run inference on, e.g., ``"cuda"`` or ``"cpu"``.
            batch_size: Number of sequences to process per batch.
            cache_dir: Directory to cache the model.
            verbose: Whether to display a progress bar.
        """
        if isinstance(model_name, ESM2Checkpoint):
            model_name = model_name.value

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer
        except ModuleNotFoundError:
            raise OptionalDependencyError("esm2") from None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)

        self.model.to(device)
        self.model.eval()

    def __call__(self, sequences: list[str]) -> np.ndarray:
        return self.embed(sequences)

    @torch.inference_mode()
    def embed(self, sequences: list[str], layer: int = -1) -> np.ndarray:
        """
        Compute embeddings of amino acid sequences.

        Each sequence is tokenized and passed through the model.
        Token embeddings are averaged (excluding special tokens) to produce a single embedding per sequence.

        Args:
            sequences: Amino acid sequences.
            layer: Layer to retrieve embeddings from.

        Returns:
            A NumPy array of shape (n_sequences, embedding_dim) containing the embeddings.
        """
        embeddings = []
        for i in tqdm(range(0, len(sequences), self.batch_size), disable=not self.verbose):
            batch = sequences[i : i + self.batch_size]
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
            tokens = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}
            hidden_state = self.model(**tokens, output_hidden_states=True).hidden_states[layer]

            lengths = [len(s) for s in batch]
            means = [hidden_state[i, 1 : length + 1].mean(dim=-2) for i, length in enumerate(lengths)]
            embed = torch.stack(means, dim=0)

            embeddings.append(embed.cpu().numpy())

        return np.concatenate(embeddings)

    @torch.inference_mode()
    def compute_pseudo_perplexity(self, sequences: list[str], mask_size: int = 1) -> np.ndarray:
        """
        Compute pseudo-perplexity for a list of sequences, masking ``mask_size`` positions per pass.

        Args:
            sequences: Amino acid sequences.
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

            logits = self.model(masked_in, attention_mask=attention_mask.to(self.device)).logits
            log_probs = torch.log_softmax(logits, dim=-1)

            for pos in pos_chunk:
                real = attention_mask[:, pos] == 1
                true_ids = input_ids[:, pos]
                pos_logps = log_probs[torch.arange(B, device=self.device), pos, true_ids]
                total_loglik[real] += pos_logps[real]

        pppls = torch.exp(-total_loglik / lengths).cpu().numpy()
        return pppls
