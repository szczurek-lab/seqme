from collections.abc import Callable

import numpy as np


class Concatenate:
    """
    Combines scalar-valued predictors into a property-based embedding.

    Each predictor maps a list of sequences to a 1D NumPy array.
    """

    def __init__(self, predictors: list[Callable[[list[str]], np.ndarray]]):
        """
        Initialize Concatenate.

        Args:
            predictors: Functions returning 1D arrays of properties.
        """
        self.predictors = predictors

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Embeds the sequences.

        Args:
            sequences: List of input sequences.

        Returns:
            Array of shape (N, D) with D properties per sequence.
        """
        properties = [predictor(sequences) for predictor in self.predictors]
        return np.stack(properties, axis=1)
