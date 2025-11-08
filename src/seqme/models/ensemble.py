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
        weights: list[float] | None = None,
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
        self.weights = weights if weights is not None else np.ones(len(predictors)) / len(predictors)

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
            Array of weighted predictions, one value per input sequence.
        """
        predictions = np.stack([pred(sequences) for pred in self.predictors], axis=1)

        if predictions.ndim != 2:
            raise ValueError(f"Expected 2 dims, got {predictions.ndim} dims.")

        weighted = predictions * self.weights
        return weighted.sum(axis=-1)
