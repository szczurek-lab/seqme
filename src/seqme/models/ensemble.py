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
        importance_weights: list[float] | None = None,
    ):
        """
        Initialize the ensemble of predictors.

        Args:
            predictors: List of callables that produce predictions for given sequences.
            importance_weights: Optional list of weights for each predictor. If omitted,
                all predictors are weighted equally.

        Raises:
            ValueError: If the length of importance_weights does not match the number of predictors.
        """
        self.predictors = predictors
        self.importance_weights = (
            importance_weights if importance_weights is not None else np.ones(len(predictors)) / len(predictors)
        )

        if len(self.importance_weights) != len(self.predictors):
            raise ValueError(
                f"importance_weights length ({len(self.importance_weights)}) must match number of predictors ({len(self.predictors)})"
            )

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Compute ensemble predictions for a list of sequences.

        Args:
            sequences: List of input strings to predict on.

        Returns:
            Array of weighted predictions, one value per input sequence.
        """
        predictions = np.stack([pred(sequences) for pred in self.predictors], axis=1)

        if predictions.ndim != 2:
            raise ValueError(f"Expected 2 dims, got {predictions.ndim} dims.")

        weighted = predictions * self.importance_weights
        return weighted.sum(axis=-1)
