from typing import Literal, get_args

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from pepme.utils.engine_cfg import EngineCfg
from pepme.utils.progress import RichProgress

ESM2_Model_OPT = Literal[
    "esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D"
]


class ESM2Embeddings:
    def __init__(
        self, model_name: Literal[ESM2_Model_OPT], device: str, batch_size: int
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
        self.model = AutoModel.from_pretrained(f"facebook/{model_name}")
        self.batch_size = batch_size
        self.device = device

    def __call__(self, sequences: list[str], *, verbose: bool = False) -> np.ndarray:
        ret = []
        with torch.inference_mode():
            for i in RichProgress(
                range(0, len(sequences), self.batch_size),
                "Computing ESM2 embeddings/BATCHES",
                verbose=verbose,
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
                for i, c in enumerate(counts):
                    mask[i, c - 1] = 0
                    mask[i, 0] = 0
                counts = counts - 2

                embed = (hidden_state * mask.unsqueeze(-1)).sum(
                    dim=-2
                ) / counts.unsqueeze(-1)
                ret.append(embed.cpu().numpy())
        return np.concatenate(ret)


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
