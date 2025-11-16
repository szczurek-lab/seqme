from enum import Enum

import numpy as np
import torch
from packaging.version import Version
from tqdm import tqdm

from .exceptions import OptionalDependencyError

_MAX_SEQUENCE_LENGTH = 512


class HyformerCheckpoint(str, Enum):
    """
    Hyformer checkpoints from Izdebski et al.

    Available checkpoints:
        molecules_8M: 8M parameters, 8 layers, embedding dim 256, pretrained on GuacaMol dataset [Brown et al.]
        molecules_50M: 50M parameters, 12 layers, embedding dim 512, pretrained on Uni-Mol dataset [Zhou et al.]
        peptides_34M: 34M parameters, 8 layers, embedding dim 512, pretrained on combined general-purpose peptide and AMP datasets [Izdebski et al.]
        peptides_34M_mic: 34M parameters, 8 layers, embedding dim 512, pretrained on combined general-purpose peptide and MIC datasets [Izdebski et al.]
            and subsequently jointly fine-tuned on peptides with (log2 transformed) MIC values against E. coli bacteria [Szymczak et al.]

    If used for prediction, pre-trained models, i.e., `molecules_8M` and `molecules_50M` and `peptides_34M`, predict the physicochemical properties used for pre-training.
    Jointly fine-tuned model `peptides_34M_mic` predicts the log2 transformed MIC values against E. coli bacteria.

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

    Installation: for molecules: ``pip install "seqme[hyformer_molecules]" "hyformer @ git+https://github.com/szczurek-lab/hyformer.git@main"``

    Installation for peptides: ``pip install "seqme[hyformer]" "hyformer @ git+https://github.com/szczurek-lab/hyformer.git@v2.0"``.

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
        cache_dir: str | None = None,
        verbose: bool = False,
    ):
        """
        Initialize Hyformer model.

        Args:
            model_name: Model checkpoint name or enum.
            device: Device to run inference on, e.g., ``"cuda"`` or ``"cpu"``.
            batch_size: Number of sequences to process per batch.
            cache_dir: Directory to cache the model.
            verbose: Whether to display a progress bar.
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
            from hyformer import __version__ as hyformer_version
            from hyformer.utils import create_dataloader
        except ModuleNotFoundError:
            raise OptionalDependencyError("hyformer") from None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_dir=cache_dir)
        self.model = AutoModel.from_pretrained(model_name, local_dir=cache_dir)

        # Hyformer version-specific attributes
        self._version = Version(hyformer_version)
        self._create_dataloader_fn = create_dataloader
        self._generative_task_key = "generation" if self._version < Version("2.0.0") else "lm"
        self._predictive_task_key = "prediction"
        self._max_sequence_length = (
            self.tokenizer.max_molecule_length if self._version < Version("2.0.0") else _MAX_SEQUENCE_LENGTH
        )
        self._logits_generation_key = "logits_generation" if self._version < Version("2.0.0") else "logits"
        self._logits_prediction_key = "logits_physchem" if self._version < Version("2.0.0") else "logits"

        self.model.to(device)
        self.model.eval()

    def __call__(self, sequences: list[str]) -> np.ndarray:
        return self.embed(sequences)

    def generate(
        self, num_samples: int, temperature: float = 1.0, top_k: int | None = None, seed: int = 0
    ) -> list[str]:
        """Generate sequences de novo.

        Delegates to the legacy generation path for Hyformer versions prior to
        2.0, otherwise uses the newer generation API.

        Args:
            num_samples: Number of sequences to produce.
            temperature: Sampling temperature passed to the decoder.
            top_k: Optional top-k sampling parameter.
            seed: Random seed forwarded to the underlying generator.

        Returns:
            A list of generated sequences, truncated to ``num_samples`` items.
        """
        if self._version < Version("2.0.0"):
            return self._generate_legacy(num_samples, temperature, top_k, seed)
        else:
            return self._generate(num_samples, temperature, top_k, seed)

    def _generate_legacy(
        self, num_samples: int, temperature: float = 1.0, top_k: int | None = None, seed: int = 0
    ) -> list[str]:
        generated = []
        for _ in tqdm(range(0, num_samples, self.batch_size), "Generating samples"):
            samples: list[str] = self.model.generate(
                self.tokenizer, min(num_samples, self.batch_size), temperature, top_k, self.device
            )
            generated.extend(self.tokenizer.decode(samples))
        return generated[:num_samples]

    def _generate(
        self, num_samples: int, temperature: float = 1.0, top_k: int | None = None, seed: int = 0
    ) -> list[str]:
        _PREFIX_INPUT_IDS = torch.tensor(
            [[self.tokenizer.task_token_id(self._generative_task_key), self.tokenizer.bos_token_id]] * self.batch_size,
            dtype=torch.long,
            device=self.device,
        )
        _USE_CACHE = False

        generated_samples = []

        with torch.inference_mode():
            for _ in tqdm(range(0, num_samples, self.batch_size), "Generating samples"):
                outputs = self.model.generate(
                    prefix_input_ids=_PREFIX_INPUT_IDS,
                    num_tokens_to_generate=self._max_sequence_length - len(_PREFIX_INPUT_IDS[0]),
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=None,
                    use_cache=_USE_CACHE,
                    seed=seed,
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
        _TASKS = {self._predictive_task_key: 1.0}

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
                batch_predictions = self.model.predict(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
                batch_predictions = (
                    batch_predictions[self._logits_prediction_key]
                    if self._version < Version("2.0.0")
                    else batch_predictions
                )
                predictions.append(batch_predictions.cpu().numpy())
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
        _TASKS = {self._predictive_task_key: 1.0}

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
        _TASKS = {self._generative_task_key: 1.0}
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
                logit_batches.append(output[self._logits_generation_key].cpu())
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

        # shift log_probs and labels by one
        log_probs = log_probs[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        ppls = torch.zeros(log_probs.shape[0])
        for idx, (log_prob, label) in enumerate(zip(log_probs, labels, strict=True)):
            ppl = 0
            n = 0
            for lp, lab in zip(log_prob, label, strict=True):
                if lab == ignore_index:
                    continue
                n += 1
                ppl += lp[lab]
            ppls[idx] = ppl / n
        ppls = torch.exp(-ppls)

        return ppls.cpu().numpy().astype(float)
