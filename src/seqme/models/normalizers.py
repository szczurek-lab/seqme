from collections.abc import Callable

import numpy as np


class MinMaxNorm:
    """
    Apply min-max normalization to the output of a predictor.

    If `min_value` and `max_value` are omitted, they are determined from the data.

    Args:
        predictor: Function taking a list of strings and returning an array of values.
        min_value: Optional fixed minimum for scaling (must set with max_value).
        max_value: Optional fixed maximum for scaling (must set with min_value).
    """

    def __init__(
        self,
        predictor: Callable[[list[str]], np.ndarray],
        min_value: float | None = None,
        max_value: float | None = None,
    ):
        """
        Initialize the normalizer.

        Args:
            predictor: Function to generate raw values.
            min_value: Explicit minimum for normalization.
            max_value: Explicit maximum for normalization.

        Raises:
            ValueError: If only one of min_value or max_value is provided.
        """
        if (min_value is None) ^ (max_value is None):
            raise ValueError("Both min_value and max_value must be set together.")

        self.predictor = predictor
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """
        Normalize predictor output to [0, 1].

        Args:
            sequences: List of input strings.

        Returns:
            Array of normalized values.
        """
        values = self.predictor(sequences)

        if (self.min_value is not None) and (self.max_value is not None):
            min_val = self.min_value
            max_val = self.max_value
        else:
            min_val = float(values.min())
            max_val = float(values.max())

        if max_val == min_val:
            return np.zeros_like(values)

        return (values - min_val) / (max_val - min_val)
