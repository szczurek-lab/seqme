from enum import Enum

import numpy as np
import torch
from tqdm import tqdm

from .exceptions import OptionalDependencyError


class GENALMCheckpoint(Enum):
    """
    GENA-LM checkpoints.

    Embedding checkpoints:
        bert_base_t2t: 110M parameters, 12 layers, embedding dim: 768, max sequence length: 512bp - Trained on T2T+1000G SNPs.
        bert_base_t2t_lastln_t2t: 110M parameters, 12 layers, embedding dim: 768, max sequence length: 512 bps - Trained on T2T+1000G SNPs.
        bert_base_t2t_multi: 110M parameters, 12 layers, embedding dim: 768, max sequence length: 512bp - Trained on T2T+1000G SNPs+Multispecies.
        bert_large_t2t: 336M parameters, 24 layers, embedding dim: 1024, max sequence length: 512bp - Trained on T2T+1000G SNPs.
        bigbird_base_t2t: 110M parameters, 12 layers, embedding dim: 768, max sequence length: 4096bp - Trained on T2T+1000G SNPs.

        Note: In practice the model has M+1 layers. The last layer is a LayerNorm.

    Downstream classification checkpoints:
        bert_base_t2t_promoters: 110M parameters, 12 layers, task sequence length: 300bp.
            Binary classification: determining the presence or absence of a promoter within a given region.

        bert_large_t2t_promoters: 336M parameters, 24 layers, task sequence length: 300bp.
            Binary classification: determining the presence or absence of a promoter within a given region.

        bert_large_t2t_promoters2: 336M parameters, 24 layers, task sequence length: 2000bp.
            Binary classification: determining the presence or absence of a promoter within a given region.

        bert_base_t2t_splice_site: 110M parameters, 12 layers, task sequence length: 15000bp. Identifies splicing sites.
            Classification: determing the splice donor, splice acceptor and none

        bert_large_t2t_splice_site: 336M parameters, 24 layers, task sequence length: 15000bp. Identifies splicing sites.
            Classification: determing the splice donor, splice acceptor and none
    """

    # Embedding
    bert_base_t2t = ("AIRI-Institute/gena-lm-bert-base-t2t", None)
    bert_base_t2t_lastln_t2t = ("AIRI-Institute/gena-lm-bert-base-lastln-t2t", None)
    bert_base_t2t_multi = ("AIRI-Institute/gena-lm-bert-base-t2t-multi", None)
    bert_large_t2t = ("AIRI-Institute/gena-lm-bert-large-t2t", None)
    bigbird_base_t2t = ("AIRI-Institute/gena-lm-bigbird-base-t2t", None)

    # Downstream classification
    bert_base_t2t_promoters = ("AIRI-Institute/gena-lm-bert-base-t2t", "promoters_300_run_1")
    bert_large_t2t_promoters = ("AIRI-Institute/gena-lm-bert-large-t2t", "promoters_300_run_1")
    bert_large_t2t_promoters2 = ("AIRI-Institute/gena-lm-bert-large-t2t", "promoters_2000_run_1")

    bert_base_t2t_splice_site = ("AIRI-Institute/gena-lm-bert-base-t2t", "spliceai_run_1")
    bert_large_t2t_splice_site = ("AIRI-Institute/gena-lm-bert-large-t2t", "spliceai_run_1")


class Task(Enum):
    EMBEDDING = "embedding"
    CLASSIFICATION = "classification"


_TASK = {
    GENALMCheckpoint.bert_base_t2t: Task.EMBEDDING,
    GENALMCheckpoint.bert_base_t2t_lastln_t2t: Task.EMBEDDING,
    GENALMCheckpoint.bert_base_t2t_multi: Task.EMBEDDING,
    GENALMCheckpoint.bert_large_t2t: Task.EMBEDDING,
    GENALMCheckpoint.bigbird_base_t2t: Task.EMBEDDING,
    GENALMCheckpoint.bert_base_t2t_promoters: Task.CLASSIFICATION,
    GENALMCheckpoint.bert_large_t2t_promoters: Task.CLASSIFICATION,
    GENALMCheckpoint.bert_large_t2t_promoters2: Task.CLASSIFICATION,
    GENALMCheckpoint.bert_base_t2t_splice_site: Task.CLASSIFICATION,
    GENALMCheckpoint.bert_large_t2t_splice_site: Task.CLASSIFICATION,
}


class GENALM:
    """
    GENA-LM is a family of Open-Source Foundational Models for Long DNA Sequences trained on human DNA sequence.

    Computes sequence-level embeddings by averaging token embeddings.

    Installation: ``pip install "seqme[genalm]"``

    Reference:
        Fishman et al., "GENA-LM: a family of open-source foundational DNA language models for long sequences"
        (https://academic.oup.com/nar/article/53/2/gkae1310/7954523)

    """

    def __init__(
        self,
        model_name: GENALMCheckpoint,
        *,
        device: str | None = None,
        batch_size: int = 256,
        cache_dir: str | None = None,
        verbose: bool = False,
    ):
        """
        Initialize model.

        Args:
            model_name: Model checkpoint name.
            device: Device to run inference on, e.g., ``"cuda"`` or ``"cpu"``.
            batch_size: Number of sequences to process per batch.
            cache_dir: Directory to cache the model.
            verbose: Whether to display a progress bar.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        try:
            from transformers import AutoModel, AutoTokenizer, BertForSequenceClassification
        except ModuleNotFoundError:
            raise OptionalDependencyError("genalm") from None

        self.task = _TASK[model_name]

        ckpt_name, branch_name = model_name.value
        self.tokenizer = AutoTokenizer.from_pretrained(ckpt_name)

        if self.task == Task.EMBEDDING:
            self.model = AutoModel.from_pretrained(ckpt_name, trust_remote_code=True, output_hidden_states=True)
        elif self.task == Task.CLASSIFICATION:
            self.model = BertForSequenceClassification.from_pretrained(
                ckpt_name, revision=branch_name, trust_remote_code=True, cache_dir=cache_dir
            )
        else:
            raise ValueError(f"Invalid task: {self.task}.")

        self.model.to(device)
        self.model.eval()

    def __call__(self, sequences: list[str]) -> np.ndarray:
        return self.embed(sequences) if self.task == Task.EMBEDDING else self.classify(sequences)

    @torch.inference_mode()
    def embed(self, sequences: list[str], layer: int = -1) -> np.ndarray:
        """
        Compute embeddings for a list of sequences.

        Each sequence is tokenized and passed through the model.
        Token embeddings are averaged to produce a single embedding per sequence.

        Args:
            sequences: List of DNA sequences.
            layer: Embedding layer.

        Returns:
            A NumPy array of shape (n_sequences, embedding_dim) containing the embeddings.
        """
        if self.task != Task.EMBEDDING:
            raise ValueError(f"Expected embedding model got {self.task} model.")

        embeddings = []
        for i in tqdm(range(0, len(sequences), self.batch_size), disable=not self.verbose):
            batch = sequences[i : i + self.batch_size]

            tokens = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=False)
            tokens = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}

            hidden_state = self.model(**tokens)["hidden_states"][layer]

            lengths = [len(s) for s in batch]
            means = [hidden_state[i, :length].mean(dim=-2) for i, length in enumerate(lengths)]
            embed = torch.stack(means, dim=0)

            embeddings.append(embed.cpu().numpy())

        return np.concatenate(embeddings)

    @torch.inference_mode()
    def classify(self, sequences: list[str]) -> np.ndarray:
        """
        Classify a list of sequences.

        Args:
            sequences: List of DNA sequences.

        Returns:
            A NumPy array of size (n_sequences, 2) for promoter prediction and (n_sequences, 3) for splice-site prediction.
        """
        if self.task != Task.CLASSIFICATION:
            raise ValueError(f"Expected classification model got {self.task} model.")

        probs = []
        for i in tqdm(range(0, len(sequences), self.batch_size), disable=not self.verbose):
            batch = sequences[i : i + self.batch_size]

            tokens = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=False, add_special_tokens=True)
            tokens = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}

            logits = self.model(**tokens)["logits"]
            batch_prob = torch.softmax(logits, dim=-1)

            probs.append(batch_prob.cpu().numpy())

        return np.concatenate(probs)
