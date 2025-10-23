import numpy as np
import torch
from tqdm import tqdm

from .exceptions import OptionalDependencyError


class ProstT5:
    """
    Wrapper for the ProstT5 encoder protein embedding model from HuggingFace.

    Computes sequence-level embeddings by averaging token embeddings.

    Checkpoint: 3B parameters, 24 layers, embedding dim 1024, trained on protein sequences and 3Di structures.

    Installation: ``pip install 'seqme[prostt5]'``

    Reference:
        Heinzinger et al., "ProstT5: Bilingual Language Model for Protein Sequence and Structure"
        (https://www.biorxiv.org/content/10.1101/2023.07.23.550085v1)
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int = 64,
        cache_dir: str | None = None,
        verbose: bool = False,
    ):
        """
        Initialize ProstT5 encoder.

        Args:
            device: Device to run inference on, e.g., "cuda" or "cpu".
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
            from transformers import T5EncoderModel, T5Tokenizer
        except ModuleNotFoundError:
            raise OptionalDependencyError("prostt5") from None

        self.tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/ProstT5",
            do_lower_case=False,
            legacy=True,
            cache_dir=cache_dir,
        )
        self.model = T5EncoderModel.from_pretrained("Rostlab/ProstT5", cache_dir=cache_dir).to(device)
        self.model.float() if device == "cpu" else self.model.half()
        self.model.eval()

    def __call__(self, sequences: list[str]) -> np.ndarray:
        return self.embed(sequences)

    @torch.inference_mode()
    def embed(self, sequences: list[str]) -> np.ndarray:
        """
        Compute embeddings for a list of sequences.

        Each sequence is tokenized and passed through the model.
        Token embeddings are averaged to produce a single embedding per sequence.

        Args:
            sequences: List of amino acid sequences.

        Returns:
            A NumPy array of shape (n_sequences, embedding_dim) containing the embeddings.
        """
        embeddings = []
        for i in tqdm(range(0, len(sequences), self.batch_size), disable=not self.verbose):
            batch = sequences[i : i + self.batch_size]
            prefixed_batch = ["<AA2fold> " + " ".join(sequence) for sequence in batch]

            tokens = self.tokenizer.batch_encode_plus(
                prefixed_batch,
                add_special_tokens=True,
                padding="longest",
                return_tensors="pt",
            ).to(self.device)

            hidden_state = self.model(tokens["input_ids"], attention_mask=tokens["attention_mask"]).last_hidden_state

            lengths = [len(s) for s in batch]
            means = [hidden_state[i, 1 : length + 1].mean(dim=-2) for i, length in enumerate(lengths)]
            embed = torch.stack(means, dim=0)

            embeddings.append(embed.cpu().numpy())

        return np.concatenate(embeddings)
