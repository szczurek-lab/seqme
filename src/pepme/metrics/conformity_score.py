from typing import Callable, Literal, Optional

import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KernelDensity

from pepme.core import Metric, MetricResult


class ConformityScore(Metric):
    """
    Computes conformity score similarly to https://prescient-design.github.io/walk-jump/
    """

    def __init__(
        self,
        reference: list[str],
        descriptors: list[Callable[[list[str]], np.ndarray]],
        n_splits: int = 5,
        kde_bandwidth: float = 0.2,
        random_state: float = 0,
        reference_name: Optional[str] = None,
    ):
        """
        Initialize the ConformityScore metric.

        Args:
            reference (list[str]): Reference sequences assumed to represent the
                target distribution.
            descriptors (list[Callable]): A list of descriptor functions. Each should
                take a list of sequences and return a 1D NumPy array of features.
            n_splits (int, optional): Number of cross-validation folds for KDE.
            kde_bandwidth (float, optional): Bandwidth parameter for the Gaussian KDE.
            random_state (float, optional): Seed for KFold shuffling.
            reference_name (str, optional): Optional name for the reference dataset.
        """
        self.reference = reference
        self.descriptors = descriptors
        self.n_splits = n_splits
        self.kde_bandwidth = kde_bandwidth
        self.random_state = random_state
        self.reference_name = reference_name

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute the conformity score for the given sequences.
        Args:
            sequences (list[str]): List of generated sequences to evaluate.

        Returns:
            MetricResult: Contains the mean and standard deviation of the conformity
                scores across all folds.
        """
        reference_arr = self._sequences_to_desc_array(
            self.reference
        )  # (n_ref, n_descs)
        generated_arr = self._sequences_to_desc_array(sequences)  # (n_gen, n_descs)

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        scores = []
        for train_idx, val_idx in kf.split(reference_arr):
            log_prob_val, log_prob_generated = self._fit_and_score_split(
                reference_arr[train_idx], reference_arr[val_idx], generated_arr
            )
            conformity_scores = self._compute_conformity_score(
                log_prob_val, log_prob_generated
            )
            scores.append(conformity_scores.mean())
        return MetricResult(float(np.mean(scores)), float(np.std(scores)))

    def _sequences_to_desc_array(self, sequences: list[str]) -> np.ndarray:
        return np.stack(
            [desc_func(sequences) for desc_func in self.descriptors], axis=1
        )

    def _fit_and_score_split(
        self, ref_train: np.ndarray, ref_val: np.ndarray, generated_arr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        kde = KernelDensity(kernel="gaussian", bandwidth=self.kde_bandwidth)
        kde.fit(ref_train)
        log_prob_val = kde.score_samples(ref_val)
        log_prob_generated = kde.score_samples(generated_arr)
        return log_prob_val, log_prob_generated

    def _compute_conformity_score(
        self, log_prob_val: np.ndarray, log_prob_generated: np.ndarray
    ) -> np.ndarray:
        n_val = log_prob_val.shape[0]
        comp_matrix = log_prob_generated[:, None] >= log_prob_val[None, :]
        counts = comp_matrix.sum(axis=1)
        conformity_scores = counts / (n_val + 1)
        return conformity_scores

    @property
    def name(self) -> str:
        return (
            "ConformityScore"
            if self.reference_name is None
            else f"ConformityScore {self.reference_name}"
        )

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        """
        #TODO - this is actually ambiguous. According to https://prescient-design.github.io/walk-jump/:
            - > 0.5: higher conformity, more similar to training data than validation data
            - 0.5: optimal conformity, as similar to training data as validation data
            - < 0.5: lower conformity, validation is more similar to training data than test data
        For now maximize
        """
        return "maximize"
