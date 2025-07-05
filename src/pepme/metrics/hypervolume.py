from typing import Callable, Literal, Optional

import numpy as np
import pymoo.indicators.hv as hv  # type: ignore
from scipy.spatial import ConvexHull, QhullError

from pepme.core import Metric, MetricResult


class Hypervolume(Metric):
    """
    Computes the hypervolume (HV) metric for multi-objective optimization.
    Each predictor maps sequences to a numeric objective.
    """

    def __init__(
        self,
        predictors: list[Callable[[list[str]], np.ndarray]],
        method: Literal["standard", "convex-hull"] = "standard",
        y_nadir: Optional[np.ndarray] = None,
        y_ideal: Optional[np.ndarray] = None,
    ):
        """
        Args:
            predictors: List of functions that output objective values for each sequence.
            method: Which HV computation method to use ("standard" or "convex-hull").
            y_nadir: Worst acceptable value in each objective dimension.
            y_ideal: Best value in each objective dimension (used for normalization).
        """
        self.predictors = predictors
        self.method = method
        self.y_nadir = y_nadir if y_nadir is not None else np.zeros(len(predictors))
        self.y_ideal = y_ideal

        if self.y_nadir.shape[0] != len(predictors):
            raise ValueError(
                f"Expected `y_nadir` to have {len(predictors)} elements, but only has {self.y_nadir.shape[0]} elements."
            )

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Evaluate hypervolume for the predicted properties of the input sequences.
        """
        values = np.stack([predictor(sequences) for predictor in self.predictors]).T
        hv_value = calculate_hypervolume(
            values,
            y_nadir=self.y_nadir,
            y_ideal=self.y_ideal,
            method=self.method,
        )
        return MetricResult(hv_value)

    @property
    def name(self) -> str:
        return "HV" if self.method == "standard" else "HV (convex-hull)"

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


def calculate_hypervolume(
    points: np.ndarray,
    y_nadir: np.ndarray,
    y_ideal: Optional[np.ndarray] = None,
    method: Literal["standard", "convex-hull"] = "standard",
) -> float:
    """
    Compute hypervolume from a set of points in objective space.

    Args:
        points: Array of shape [N, D] with objective values.
        y_nadir: Reference point (worse than or equal to all actual points).
        y_ideal: Optional ideal point for normalization.
        method: Either "standard" using pymoo, or "convex-hull" using scipy.

    Returns:
        Hypervolume (float)
    """
    if points.shape[1] != y_nadir.shape[0]:
        raise ValueError(
            "Points must have the same number of dimensions as the reference point."
        )

    min_points = points.min(axis=0)
    if (y_nadir > min_points).any():
        raise ValueError(
            f"Invalid `y_nadir`: each component must be less than or equal to the minimum value in that dimension.\n"
            f"Provided `y_nadir`: {y_nadir}\n"
            f"Minimum required values: {min_points}"
        )

    if points.shape[0] <= 1:
        return float("nan")

    points = points.copy() - y_nadir
    ref_point = np.zeros(points.shape[1])

    if y_ideal is not None:
        points = points / (y_ideal - y_nadir)

    # Compute hypervolume using selected method
    if method == "standard":
        hv_indicator = hv.HV(ref_point=ref_point)
        hypervolume = hv_indicator(-points).item()
    elif method == "convex-hull":
        all_points = np.vstack((points, ref_point))
        try:
            hypervolume = ConvexHull(all_points).volume
        except QhullError:
            hypervolume = float("nan")  # Return NaN if hull can't be formed
    else:
        raise ValueError(f"Unknown method '{method}'.")

    return hypervolume


class HV(Hypervolume):
    pass
