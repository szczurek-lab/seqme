import warnings

import numpy as np

from typing import Literal
from collections.abc import Callable

from pepme.core import Metric, MetricResult
from pepme.gmm import FrozenMeanGMM


class FLD(Metric):
    """ Computes the Feature Likelihood Divergence (FLD) metric.

    Reference:
        Jiralerspong et al., "Feature Likelihood Divergence: Evaluating the
        Generalization of Generative Models Using Samples" (https://arxiv.org/pdf/2302.04440)
    """

    def __init__(
        self,
        reference: list[str] | dict[list[str], list[str]],
        embedder: Callable[[list[str]], np.ndarray],
        reference_name: dict[str] | str | None = None,
        embedder_name: str | None = None,
        random_state: int | None = None,
    ):
        """
        Initializes the FLD metric with a reference train and test datasets.

        Args:
            reference: A list of reference sequences (e.g., real data).
            embedder: A function that maps a list of sequences to a 2D NumPy array of embeddings.
            reference_name: Optional name for the reference dataset.
            embedder_name: Optional name for the embedder used.

        Raises:
            ValueError: If fewer than 2 reference embeddings are provided.

        Example:
            >>> fld = FLD(
            ...     reference={
            ...         "train": train_sequences,
            ...         "test": test_sequences
            ...     },
            ...     embedder=embedder,
            ...     reference_name={
            ...         "train": "train",
            ...         "test": "test"
            ...     },
            ...     embedder_name="Hyformer",
            ...     random_state=42
            ... )
            >>> fld_score = fld(generated_sequences)
            >>> print(fld_score)

        """
        self.reference = reference
        self.embedder = embedder
        self.reference_name = reference_name
        self.embedder_name = embedder_name
        self.random_state = random_state
        self.reference_embeddings = self.embedder(self.reference)

        if not isinstance(self.reference, dict):
            raise ValueError("Reference must be a dictionary with train and test datasets")
        
        assert "train" in self.reference, "Reference must contain train dataset"
        assert "test" in self.reference, "Reference must contain test dataset"
        
        if self.reference_embeddings["train"].shape[0] < 2:
            raise ValueError("Reference embeddings must contain at least two samples.")
        
        if self.reference_embeddings["test"].shape[0] < 2:
            raise ValueError("Reference embeddings must contain at least two samples.")

    def __call__(self, sequences: list[str], normalize_embeddings: bool = True) -> MetricResult:
        """
        Computes the FLD between the reference and the input sequences.

        Args:
            sequences: A list of generated sequences to evaluate.

        Returns:
            MetricResult containing the FID score. Lower is better.
        """
        sequence_embeddings = self.embedder(sequences)
        if normalize_embeddings:
            sequence_embeddings, sequence_embeddings_train, sequence_embeddings_test = _preprocess(
                sequence_embeddings,
                self.reference_embeddings["train"],
                self.reference_embeddings["test"]
            )
        else:
            sequence_embeddings_train, sequence_embeddings_test = self.reference_embeddings["train"], self.reference_embeddings["test"]
            
        gmm = FrozenMeanGMM(init_means=sequence_embeddings, random_state=self.random_state).fit(sequence_embeddings_train)
        if not gmm.converged_:
            warnings.warn("GMM did not converge")
        log_likelihood = gmm.compute_log_likelihood(sequence_embeddings_test)
        fld = self._compute_normalized_nll(log_likelihood)
        fld = fld.mean().item()
        fld_baseline = self._compute_baseline_nll(sequences).mean().item()
        fld = (fld - fld_baseline) * 100
        return MetricResult(fld)

    @staticmethod
    def _compute_normalized_nll(log_likelihood):
        return (-1) * log_likelihood / log_likelihood.shape[1]
    
    def _compute_baseline_nll(self, sequences):
        sequence_embeddings = self.embedder(sequences)
        n = len(sequence_embeddings)
        train_embeddings, test_embeddings, _ = _preprocess(
            self.reference_embeddings["train"],
            self.reference_embeddings["test"],
            sequence_embeddings
        )
        train_embeddings = _shuffle(train_embeddings)

        split_size = n // 2
        gmm = FrozenMeanGMM(train_embeddings[:split_size]).fit(train_embeddings[split_size:])
        if not gmm.converged_:
            warnings.warn("GMM did not converge")
        baseline_nll = gmm.compute_log_likelihood(test_embeddings)
        return self._compute_normalized_nll(baseline_nll)

    @property
    def name(self) -> str:
        name = "FLD"
        if self.embedder_name:
            name += f"@{self.embedder_name}"
        if self.reference_name:
            name += f" ({self.reference_name})"
        return name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "minimize"


def _preprocess(train_x, test_x, gen_x, normalize=True):
    mean_vals, std_vals = test_x.mean(axis=0), test_x.std(axis=0)

    def normalize_feat(feat):
        feat = (feat - mean_vals) / (std_vals + 1e-8)  # Add small epsilon to avoid division by zero
        return feat

    if normalize:
        train_x = normalize_feat(train_x)
        test_x = normalize_feat(test_x)
        gen_x = normalize_feat(gen_x)

    return train_x, test_x, gen_x


def _shuffle(x, size=None):
    if size is not None:
        size = min(size, len(x))
    idx = np.random.choice(len(x), size if size else len(x), replace=False)
    return x[idx]
