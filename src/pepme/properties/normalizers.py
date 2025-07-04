from typing import Callable, Optional

import numpy as np


class MinMaxNorm:
    def __init__(
        self,
        predictor: Callable[[list[str]], np.ndarray],
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ):
        self.predictor = predictor

        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, sequences: list[str]) -> np.ndarray:
        values = self.predictor(sequences)

        if (self.min_value is not None) or (self.max_value is not None):
            if None in (self.min_value, self.max_value):
                raise ValueError(
                    "Both min_value and max_value must be set if one is set."
                )

            min_value = self.min_value
            max_value = self.max_value
        else:
            min_value = values.min()
            max_value = values.max()

        return (values - min_value) / (max_value - min_value)
