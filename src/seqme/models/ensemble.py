from collections.abc import Callable

import numpy as np


class Ensemble:
    """
    Combines multiple predictor functions into a weighted ensemble.

    Each predictor maps sequences to numeric arrays. The final output is a
    weighted sum of individual predictions.
    """

    def __init__(
        self,
        predictors: list[Callable[[list[str]], np.ndarray]],
        weights: list[float] | np.ndarray | None = None,
    ):
        """
        Initialize the ensemble of predictors.

        Args:
            predictors: List of callables that produce predictions for the given sequences.
            weights: Optional list of weights for each predictor. If ``None``, all predictors are weighted equally.

        Raises:
            ValueError: If the length of ``weights`` does not match the number of predictors.
        """
        self.predictors = predictors

        if weights is None:
            self.weights = np.ones(len(predictors), dtype=np.float64)
        else:
            self.weights = np.asarray(weights, dtype=np.float64)

        self.weights /= self.weights.sum()

        if len(self.weights) != len(self.predictors):
            raise ValueError(
                f"weights length ({len(self.weights)}) must match number of predictors ({len(self.predictors)})"
            )

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Compute ensemble predictions on a list of sequences.

        Args:
            sequences: Input sequences to the predictors.

        Returns:
            Array of weighted predictions.
        """
        predictions = np.stack([pred(sequences) for pred in self.predictors], axis=1)

        if predictions.ndim < 2:
            raise ValueError(f"Expected at least 2 dims, got {predictions.ndim} dims.")

        weighted_predictions = np.einsum("sp...,p->s...", predictions, self.weights)

        return weighted_predictions
