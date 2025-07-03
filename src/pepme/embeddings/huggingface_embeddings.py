from typing import Literal, get_args

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from transformers.utils import logging

from pepme.utils.engine_cfg import EngineCfg
from pepme.utils.progress import RichProgress

ESM2_Model_OPT = Literal[
    "esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D"
]


class ESM2Embeddings:
    def __init__(
        self,
        model_name: Literal[ESM2_Model_OPT],
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
            for i in RichProgress(
                range(0, len(sequences), self.batch_size),
                "Computing ESM2 embeddings/BATCHES",
                verbose=self.verbose,
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


HuggingFaceModel_OPT = Literal[ESM2_Model_OPT,]


def compute_huggingface_model_embeddings(
    sequences: list[str],
    opt: HuggingFaceModel_OPT,
    *,
    device: str | None = None,
    batch_size: int | None = None,
) -> np.ndarray:
    if device is None:
        device = EngineCfg.DEVICE().type
    if batch_size is None:
        batch_size = EngineCfg.BATCH_DIM()

    if opt in get_args(ESM2_Model_OPT):
        encode_fn = ESM2Embeddings(
            model_name=opt,
            device=device,
            batch_size=batch_size,
        )
    else:
        raise NotImplementedError

    return encode_fn(sequences)
