import numpy as np
import torch
from tqdm import tqdm

from .exceptions import OptionalDependencyError


class DNABert2:
    """
    Wrapper for the DNABert2 embedding model from HuggingFace.

    Computes sequence-level embeddings by averaging token embeddings.

    Installation: ``pip install seqme[DNABert2]``

    Reference:
        Zhou et al., "DNABERT-2: Efficient Foundation Model and Benchmark For Multi-Species Genome"
        (https://arxiv.org/abs/2306.15006)

    """

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int = 256,
        verbose: bool = False,
    ):
        """
        Initialize model.

        Args:
            device: Device to run inference on, e.g., "cuda" or "cpu".
            batch_size: Number of sequences to process per batch.
            verbose: Whether to display a progress bar.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        try:
            from transformers import AutoModel, AutoTokenizer
        except ModuleNotFoundError:
            raise OptionalDependencyError("DNABert2") from None

        self.tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
        self.model = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

        self.model.to(device)
        self.model.eval()

    def __call__(self, sequences: list[str]) -> np.ndarray:
        return self.embed(sequences)

    def embed(self, sequences: list[str]) -> np.ndarray:
        """
        Compute embeddings for a list of sequences.

        Each sequence is tokenized and passed through the model.
        Token embeddings are averaged to produce a single embedding per sequence.

        Args:
            sequences: List of DNA sequences.

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

                hidden_state = self.model(**tokens)[0]

                lengths = [len(s) for s in batch]
                means = [hidden_state[i, :length].mean(dim=-2) for i, length in enumerate(lengths)]
                embed = torch.stack(means, dim=0)

                embeddings.append(embed.cpu().numpy())

        return np.concatenate(embeddings)
