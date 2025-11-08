from collections.abc import Callable
from typing import Literal

import moocore
import numpy as np
from scipy.spatial import ConvexHull, QhullError

from seqme.core.base import Metric, MetricResult


class Hypervolume(Metric):
    """Hypervolume metric for multi-objective optimization."""

    def __init__(
        self,
        predictors: list[Callable[[list[str]], np.ndarray]],
        *,
        method: Literal["hvi", "convex-hull"] = "hvi",
        nadir: np.ndarray | None = None,
        ideal: np.ndarray | None = None,
        strict: bool = True,
        name: str = "Hypervolume",
    ):
        """
        Initialize the metric.

        Args:
            predictors: A list of functions. Each function maps a sequence to a numeric value aimed to be maximized.
            method: Which Hypervolume computation method to use

                - ``'hvi'``: Hypervolume indicator
                - ``'convex-hull'``: Volume of the convex-hull

            nadir: Smallest (worst) value in each objective dimension. If ``None``, set to zero vector.
            ideal: Largest (best) value in each objective dimension (used for normalizing points to [0;1]).
            strict: If ``True`` and values < ``nadir`` (or values > ``ideal``) raise an exception.
            name: Metric name.
        """
        self.predictors = predictors
        self.method = method
        self.nadir = nadir if nadir is not None else np.zeros(len(predictors))
        self.ideal = ideal
        self.strict = strict
        self._name = name

        if self.nadir.shape[0] != len(predictors):
            raise ValueError(
                f"Expected nadir to have {len(predictors)} elements, but only has {self.nadir.shape[0]} elements."
            )

        if self.ideal is not None:
            if self.ideal.shape[0] != len(predictors):
                raise ValueError(
                    f"Expected ideal to have {len(predictors)} elements, but only has {self.ideal.shape[0]} elements."
                )

            if (self.ideal < self.nadir).any():
                raise ValueError("Expected nadir <= ideal.")

    def __call__(self, sequences: list[str]) -> MetricResult:
        """Compute hypervolume for the predicted properties of the input sequences.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: Hypervolume.
        """
        values = np.stack([predictor(sequences) for predictor in self.predictors], axis=1)
        hypervolume = calculate_hypervolume(values, self.nadir, self.ideal, self.method, self.strict)
        return MetricResult(hypervolume)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


def calculate_hypervolume(
    points: np.ndarray,
    nadir: np.ndarray,
    ideal: np.ndarray | None = None,
    method: Literal["hvi", "convex-hull"] = "hvi",
    strict: bool = True,
) -> float:
    """
    Compute hypervolume from a set of points in objective space.

    Args:
        points: Array of shape [N, D] with objective values.
        nadir: Reference point (worse than or equal to all actual points).
        ideal: Best value in each objective dimension (used for normalizing points to [0;1]).
        method: Either hypervolume indicator ("hvi") or "convex-hull".
        strict: If ``True``, if values < ``nadir`` (or values > ``ideal``) raise an exception.

    Returns:
        Hypervolume
    """
    if points.shape[1] != nadir.shape[0]:
        raise ValueError("Points must have the same number of dimensions as the reference point.")

    # replace NaN values with nadir
    points = np.where(np.isnan(points), nadir, points)

    if strict:
        min_elements = points.min(axis=0)
        if (nadir > min_elements).any():
            raise ValueError(f"Value smaller than nadir. Point: {min_elements}. nadir: {nadir}")

        if ideal is not None:
            max_elements = points.max(axis=0)
            if (ideal < max_elements).any():
                raise ValueError(f"Value larger than ideal. Point: {max_elements}. ideal: {ideal}")

    points = np.maximum(points, nadir)
    if ideal is not None:
        points = np.minimum(points, ideal)

    points = points - nadir
    ref_point = np.zeros(points.shape[1])

    if ideal is not None:
        points = points / (ideal - nadir)

    if method == "hvi":
        hypervolume = moocore.hypervolume(points, ref=ref_point, maximise=True)
    elif method == "convex-hull":
        all_points = np.vstack((points, ref_point))
        try:
            hypervolume = ConvexHull(all_points).volume
        except QhullError:
            hypervolume = float("nan")  # Return NaN if hull can't be formed
    else:
        raise ValueError(f"Unknown method: {method}")

    return hypervolume
