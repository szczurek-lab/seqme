from enum import Enum

import numpy as np
import torch
from tqdm import tqdm

from .exceptions import OptionalDependencyError


class HyformerCheckpoint(str, Enum):
    """
    Hyformer checkpoints from Izdebski et al.

    Available checkpoints:
        molecules_8M: 8M parameters, 8 layers, embedding dim 256, pretrained on GuacaMol dataset [Brown et al.]
        molecules_50M: 50M parameters, 12 layers, embedding dim 512, pretrained on Uni-Mol dataset [Zhou et al.]
        peptides_34M: 34M parameters, 8 layers, embedding dim 512, pretrained on combined general-purpose peptide and AMP datasets [Izdebski et al.]
        peptides_34M_mic: 34M parameters, 8 layers, embedding dim 512, pretrained on combined general-purpose peptide and MIC datasets [Izdebski et al.]
            and subsequently jointly fine-tuned on peptides with MIC values against E. coli bacteria [Szymczak et al.]

    Reference:
        Izdebski et al. "Synergistic Benefits of Joint Molecule Generation and Property Prediction"
        Brown et al. "GuacaMol: benchmarking models for de novo molecular design"
        Zhou et al. "Uni-mol: A universal 3d molecular representation learning framework"
        Szymczak et al. "Discovering highly potent antimicrobial peptides with deep generative model hydramp"
    """

    # molecules checkpoints
    molecules_8M = "SzczurekLab/hyformer_molecules_8M"
    molecules_50M = "SzczurekLab/hyformer_molecules_50M"

    # peptides checkpoints
    peptides_34M = "SzczurekLab/hyformer_peptides_34M"
    peptides_34M_mic = "SzczurekLab/hyformer_peptides_34M_mic"


class Hyformer:
    """
    Wrapper for the Hyformer molecule/peptide embedding model.

    Computes sequence-level embeddings by extracting the [CLS] token embedding.

    Installation: ``pip install seqme[hyformer]``

    Reference:
        Izdebski et al., "Synergistic Benefits of Joint Molecule Generation and Property Prediction"
        (https://arxiv.org/abs/2504.16559)
    """

    def __init__(
        self,
        model_name: HyformerCheckpoint | str,
        *,
        device: str | None = None,
        batch_size: int = 256,
        verbose: bool = False,
        cache_dir: str | None = None,
    ):
        """
        Initialize Hyformer model.

        Args:
            model_name: Model checkpoint name or enum.
            device: Device to run inference on, e.g., "cuda" or "cpu".
            batch_size: Number of sequences to process per batch.
            verbose: Whether to display a progress bar.
            cache_dir: Directory to cache the model.
        """
        if isinstance(model_name, HyformerCheckpoint):
            model_name = model_name.value

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        try:
            from hyformer import AutoModel, AutoTokenizer
            from hyformer.utils import create_dataloader
        except ModuleNotFoundError:
            raise OptionalDependencyError("hyformer") from None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, local_dir=cache_dir)
        self._create_dataloader_fn = create_dataloader

        self.model.to(device)
        self.model.eval()

    def __call__(self, sequences: list[str]) -> np.ndarray:
        return self.embed(sequences)

    def generate(self, num_samples: int, temperature: float = 1.0, top_k: int | None = None, seed: int = 1337) -> list[str]:
        _MAX_SEQUENCE_LENGTH = 256
        _PREFIX_INPUT_IDS = torch.tensor(
            [[self.tokenizer.task_token_id("lm"), self.tokenizer.bos_token_id]] * self.batch_size,
            dtype=torch.long,
            device=self.device,
        )
        _USE_CACHE = False

        generated_samples = []

        with torch.inference_mode():
            for _ in tqdm(range(0, num_samples, self.batch_size), "Generating samples"):
                outputs = self.model.generate(
                    prefix_input_ids=_PREFIX_INPUT_IDS,
                    num_tokens_to_generate=_MAX_SEQUENCE_LENGTH - len(_PREFIX_INPUT_IDS[0]),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=None,
                    use_cache=_USE_CACHE,
                    seed=seed
                )
                generated_samples.extend(self.tokenizer.decode(outputs))

        return generated_samples[:num_samples]

    def predict(self, sequences: list[str]) -> np.ndarray:
        """
        Compute predictions for a list of sequences.

        Each sequence is tokenized and passed through the model.
        Token predictions are [CLS] token predictions.

        Args:
            sequences: List of input sequences.

        Returns:
            A NumPy array of shape (n_sequences, num_prediction_tasks) containing the predictions.
        """
        _CLS_TOKEN_IDX = 0
        _TASKS = {"prediction": 1.0}

        _dataloader = self._create_dataloader_fn(
            dataset=sequences,
            tasks=_TASKS,
            tokenizer=self.tokenizer,
            batch_size=min(len(sequences), self.batch_size),
            shuffle=False,
        )

        predictions = []
        with torch.inference_mode():
            for batch in tqdm(
                _dataloader,
                disable=not self.verbose,
            ):
                batch = batch.to_device(self.device)
                batch_predictions = (
                    self.model.predict(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    .cpu()
                    .numpy()
                )
                predictions.append(batch_predictions)
        return np.concatenate(predictions, axis=0)

    def embed(self, sequences: list[str]) -> np.ndarray:
        """
        Compute embeddings for a list of sequences.

        Each sequence is tokenized and passed through the model.
        Token embeddings are [CLS] token embeddings.

        Args:
            sequences: List of input amino acid sequences.

        Returns:
            A NumPy array of shape (n_sequences, embedding_dim) containing the embeddings.
        """
        _CLS_TOKEN_IDX = 0
        _TASKS = {"prediction": 1.0}

        _dataloader = self._create_dataloader_fn(
            dataset=sequences,
            tasks=_TASKS,
            tokenizer=self.tokenizer,
            batch_size=min(len(sequences), self.batch_size),
            shuffle=False,
        )

        embeddings = []
        with torch.inference_mode():
            for batch in tqdm(
                _dataloader,
                disable=not self.verbose,
            ):
                batch = batch.to_device(self.device)
                output = self.model(**batch, return_loss=False)
                batch_embeddings = output["embeddings"][:, _CLS_TOKEN_IDX].detach().cpu().numpy()
                embeddings.append(batch_embeddings)
        return np.concatenate(embeddings, axis=0)

    def compute_perplexity(self, sequences: list[str]) -> np.ndarray:
        """
        Compute perplexity for a list of sequences.

        Args:
            sequences: List of sequences.

        Returns:
            np.ndarray: Perplexity scores, in the same order as the input sequences.
        """
        _TASKS = {"lm": 1.0}
        _dataloader = self._create_dataloader_fn(
            dataset=sequences,
            tasks=_TASKS,
            tokenizer=self.tokenizer,
            batch_size=min(len(sequences), self.batch_size),
            shuffle=False,
        )

        logit_batches: list[torch.Tensor] = []
        label_batches: list[torch.Tensor] = []

        with torch.inference_mode():
            for batch in tqdm(
                _dataloader,
                disable=not self.verbose,
            ):
                batch = batch.to_device(self.device)
                output = self.model(**batch, return_loss=False)
                logit_batches.append(output["logits"].cpu())
                label_batches.append(batch["input_labels"].cpu())

        logits = torch.cat(logit_batches, dim=0)
        labels = torch.cat(label_batches, dim=0)

        return self._perplexity_from_logits(logits, labels)

    @staticmethod
    def _perplexity_from_logits(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> np.ndarray:
        """Compute sequence-level perplexity from token logits.

        Args:
            logits: Float tensor of shape (batch, seq_len, vocab_size) with unnormalized scores.
            labels: Long tensor of shape (batch, seq_len) with token ids used as targets.
            ignore_index: Index to ignore in the labels.

        Returns:
            Array of shape (batch,) with perplexity per sequence.
        """
        if logits.ndim != 3:
            raise ValueError("logits must have shape (batch, seq_len, vocab_size)")
        if labels.ndim != 2:
            raise ValueError("labels must have shape (batch, seq_len)")
        if labels.shape[:2] != logits.shape[:2]:
            raise ValueError("labels and logits must share (batch, seq_len)")

        # log-softmax over the vocabulary for numerical stability
        log_probs = torch.log_softmax(logits, dim=-1)  # (batch, seq_len, vocab)

        # shift logits and labels by one
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        ppls = torch.zeros(logits.shape[0])
        for idx, (log_prob, label) in enumerate(zip(log_probs, labels, strict=False)):
            ppl = 0
            n = 0
            for lp, lab in zip(log_prob, label, strict=False):
                if lab == ignore_index:
                    continue
                n += 1
                ppl += lp[lab]
            ppls[idx] = ppl / n
        ppls = torch.exp(-ppls)

        return ppls.cpu().numpy().astype(float)
