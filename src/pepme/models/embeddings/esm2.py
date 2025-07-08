from typing import Literal

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging


class ESM2:
    """Computes ESM2 embeddings."""

    def __init__(
        self,
        model_name: Literal[
            "esm2_t6_8M_UR50D",
            "esm2_t12_35M_UR50D",
            "esm2_t30_150M_UR50D",
            "esm2_t33_650M_UR50D",
            "esm2_t36_3B_UR50D",
            "esm2_t48_15B_UR50D",
        ],
        device: str,
        batch_size: int,
        verbose: bool = False,
    ):
        v = logging.get_verbosity()
        logging.set_verbosity_error()
        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
        self.model = AutoModel.from_pretrained(f"facebook/{model_name}")
        logging.set_verbosity(v)

        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

    def __call__(self, sequences: list[str]) -> np.ndarray:
        embeddings = []
        with torch.inference_mode():
            for i in tqdm(
                range(0, len(sequences), self.batch_size),
                disable=not self.verbose,
            ):
                batch = sequences[i : i + self.batch_size]
                tokens = self.tokenizer(
                    batch, return_tensors="pt", padding=True, truncation=False
                )
                tokens = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in tokens.items()
                }
                hidden_state = self.model(**tokens).last_hidden_state

                counts = tokens["attention_mask"].sum(dim=-1)
                mask = tokens["attention_mask"]

                batch_size = mask.size(0)
                batch_indices = torch.arange(batch_size, device=mask.device)
                mask[batch_indices, 0] = 0
                mask[batch_indices, counts - 1] = 0
                counts = counts - 2

                embed = (hidden_state * mask.unsqueeze(-1)).sum(
                    dim=-2
                ) / counts.unsqueeze(-1)
                embeddings.append(embed.cpu().numpy())
        return np.concatenate(embeddings)
