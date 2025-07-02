from functools import partial
from typing import Literal, TypeVar, get_args

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from pepme.utils.engine_cfg import EngineCfg
from pepme.utils.progress import RichProgress

ESM2_Model_OPT = Literal[
    "esm2_t6_8M_UR50D", "esm2_t12_35M_UR50D", "esm2_t30_150M_UR50D"
]


def compute_esm2_embeddings(
    proteins: list[str],
    device: str,
    batch_size: int,
    model_name: Literal[ESM2_Model_OPT],
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(f"facebook/{model_name}")
    model = AutoModel.from_pretrained(f"facebook/{model_name}")
    ret = []
    with torch.inference_mode():
        for i in RichProgress(
            range(0, len(proteins), batch_size), "Computing ESM2 embeddings/BATCHES"
        ):
            batch = proteins[i : i + batch_size]
            tokens = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=False
            )
            tokens = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in tokens.items()
            }
            hidden_state = model(**tokens).last_hidden_state

            counts = tokens["attention_mask"].sum(dim=-1)
            mask = tokens["attention_mask"]
            for i, c in enumerate(counts):
                mask[i, c - 1] = 0
                mask[i, 0] = 0
            counts = counts - 2

            embed = (hidden_state * mask.unsqueeze(-1)).sum(dim=-2) / counts.unsqueeze(
                -1
            )
            ret.append(embed.cpu().numpy())
    return np.concatenate(ret)


HuggingFaceModel_OPT = Literal[ESM2_Model_OPT,]

K = TypeVar("K")


def compute_huggingface_model_embeddings(
    proteins: list[str],
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
        encode_fn = partial(
            compute_esm2_embeddings,
            device=device,
            batch_size=batch_size,
            model_name=opt,
        )
    else:
        raise NotImplementedError

    return encode_fn(proteins)
