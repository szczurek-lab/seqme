from collections.abc import Callable
from typing import Literal

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
        nadir: np.ndarray | None = None,
        ideal: np.ndarray | None = None,
        include_objective_count_in_name: bool = True,
    ):
        """
        Args:
            predictors: List of functions that output objective values for each sequence.
            method: Which HV computation method to use ("standard" or "convex-hull").
            nadir: Worst acceptable value in each objective dimension.
            ideal: Best value in each objective dimension (used for normalization).
            include_objective_count_in_name: Whether to append the number of objectives to the metric's name.
        """
        self.predictors = predictors
        self.method = method
        self.nadir = nadir if nadir is not None else np.zeros(len(predictors))
        self.ideal = ideal
        self.include_objective_count_in_name = include_objective_count_in_name

        if self.nadir.shape[0] != len(predictors):
            raise ValueError(
                f"Expected `nadir` to have {len(predictors)} elements, but only has {self.nadir.shape[0]} elements."
            )

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Evaluate hypervolume for the predicted properties of the input sequences.
        """
        values = np.stack([predictor(sequences) for predictor in self.predictors]).T
        hv_value = calculate_hypervolume(
            values,
            nadir=self.nadir,
            ideal=self.ideal,
            method=self.method,
        )
        return MetricResult(hv_value)

    @property
    def name(self) -> str:
        name = "HV"
        if self.include_objective_count_in_name:
            name += f"-{len(self.predictors)}"
        if self.method == "convex-hull":
            name += " (convex-hull)"
        return name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


def calculate_hypervolume(
    points: np.ndarray,
    nadir: np.ndarray,
    ideal: np.ndarray | None = None,
    method: Literal["standard", "convex-hull"] = "standard",
) -> float:
    """
    Compute hypervolume from a set of points in objective space.

    Args:
        points: Array of shape [N, D] with objective values.
        nadir: Reference point (worse than or equal to all actual points).
        ideal: Optional ideal point for normalization.
        method: Either "standard" using pymoo, or "convex-hull" using scipy.

    Returns:
        Hypervolume (float)
    """
    if points.shape[1] != nadir.shape[0]:
        raise ValueError("Points must have the same number of dimensions as the reference point.")

    min_points = points.min(axis=0)
    if (nadir > min_points).any():
        raise ValueError(
            f"Invalid `nadir`: each component must be less than or equal to the minimum value in that dimension.\n"
            f"Provided `nadir`: {nadir}\n"
            f"Minimum required values: {min_points}"
        )

    if points.shape[0] <= 1:
        return float("nan")

    points = points.copy() - nadir
    ref_point = np.zeros(points.shape[1])

    if ideal is not None:
        points = points / (ideal - nadir)

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
    """
    Computes the hypervolume (HV) metric for multi-objective optimization.
    Each predictor maps sequences to a numeric objective.
    """

    pass
