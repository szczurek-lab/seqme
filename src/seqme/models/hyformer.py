from enum import Enum
 
import numpy as np
import torch
from tqdm import tqdm

from .exceptions import OptionalDependencyError


class HyformerCheckpoint(str, Enum):
    # molecules checkpoints
    molecules_8M = "SzczurekLab/hyformer_molecules_8M"
    molecules_50M = "SzczurekLab/hyformer_molecules_50M"
    
    # peptides checkpoints
    peptides = "SzczurekLab/hyformer_peptides"
    peptides_mic = "SzczurekLab/hyformer_peptides_mic"
    

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
        seed: int = 1337,
    ):
        """
        Initialize Hyformer model.

        Args:
            model_name: Model checkpoint name or enum.
            device: Device to run inference on, e.g., "cuda" or "cpu".
            batch_size: Number of sequences to process per batch.
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
            from hyformer import AutoTokenizer, AutoModel
            from hyformer.utils import create_dataloader
        except ModuleNotFoundError:
            raise OptionalDependencyError("hyformer") from None

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self._create_dataloader_fn = create_dataloader

        self.model.to(device)
        self.model.eval()

    def __call__(self, sequences: list[str]) -> np.ndarray:
        return self.embed(sequences)

    def generate(self, sequences: list[str]) -> np.ndarray:
        pass

    def embed(self, sequences: list[str]) -> np.ndarray:
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

        logits = []
        labels = []
        
        with torch.inference_mode():
            for batch in tqdm(
                _dataloader,
                disable=not self.verbose,
            ):
                batch = batch.to_device(self.device)
                output = self.model(**batch, return_loss=False)
                logits.append(output["logits"].cpu())
                labels.append(batch["input_labels"].cpu())
        
        logits = torch.cat(logits, dim=0)
        labels = torch.cat(labels, dim=0)

        return self._perplexity_from_logits(logits, labels)

    @staticmethod
    def _perplexity_from_logits(logits: "torch.Tensor", labels: "torch.Tensor", ignore_index: int = -100) -> np.ndarray:
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
        for idx, (log_prob, label) in enumerate(zip(log_probs, labels)):
            ppl = 0
            n = 0
            for lp, lab in zip(log_prob, label):
                if lab == ignore_index:
                    continue
                n += 1
                ppl += lp[lab]
            ppls[idx] = ppl / n
        ppls = torch.exp(-ppls)

        return ppls.cpu().numpy().astype(float)
