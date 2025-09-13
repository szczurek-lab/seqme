import numpy as np
import torch
from tqdm import tqdm


class ProstT5:
    """
    Wrapper for the ProstT5 encoder protein embedding model from HuggingFace.

    Computes sequence-level embeddings by averaging token embeddings.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int = 32,
        verbose: bool = False,
    ):
        """
        Initialize ProstT5 encoder.

        Args:
            model_name: Model checkpoint name or enum.
            device: Device to run inference on, e.g., "cuda" or "cpu".
            batch_size: Number of sequences to process per batch.
            verbose: Whether to display a progress bar.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        from transformers import T5EncoderModel, T5Tokenizer

        self.tokenizer = T5Tokenizer.from_pretrained("Rostlab/ProstT5", do_lower_case=False, legacy=True)
        self.model = T5EncoderModel.from_pretrained("Rostlab/ProstT5").to(device)
        self.model.float() if device == "cpu" else self.model.half()

        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

    def __call__(self, sequences: list[str]) -> np.ndarray:
        return self.embed(sequences)

    def embed(self, sequences: list[str]) -> np.ndarray:
        """
        Compute embeddings for a list of sequences.

        Each sequence is tokenized and passed through the model.
        Token embeddings are averaged to produce a single embedding per sequence.

        Args:
            sequences: List of input amino acid sequences.

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
                batch = ["<AA2fold> " + " ".join(sequence) for sequence in batch]

                tokens = self.tokenizer.batch_encode_plus(
                    batch,
                    add_special_tokens=True,
                    padding="longest",
                    return_tensors="pt",
                ).to(self.device)

                hidden_state = self.model(
                    tokens["input_ids"], attention_mask=tokens["attention_mask"]
                ).last_hidden_state

                counts = tokens["attention_mask"].sum(dim=-1)
                mask = tokens["attention_mask"]

                embed = (hidden_state * mask.unsqueeze(-1)).sum(dim=-2) / counts.unsqueeze(-1)
                embeddings.append(embed.cpu().numpy())

        return np.concatenate(embeddings)
