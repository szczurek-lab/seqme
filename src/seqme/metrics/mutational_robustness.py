from typing import Literal, Callable

import numpy as np
import random
from seqme.core.base import Metric, MetricResult


class MutationalRobustness(Metric):
    """Assess stability of generated biological sequences under point mutations."""

    def __init__(
        self,
        predictor: Callable[[list[str]], np.ndarray],
        *,
        n_mutations: int = 3,
        alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
        seed: int = 0,
        name: str = "MutationalRobustness",
    ):
        """
        Initialize the metric.

        Args:
            predictor: Function or model that maps a list of sequences to an array of predicted numeric scores.
            n_mutations: Number of random point mutations to apply per sequence.
            alphabet: Allowed set of mutation characters (default: 20 amino acids).
            seed: Random seed for deterministic mutations.
            name: Metric name.
        """
        if n_mutations <= 0:
            raise ValueError("Expected n_mutations > 0.")
        if not alphabet:
            raise ValueError("Alphabet cannot be empty.")

        self.predictor = predictor
        self.n_mutations = n_mutations
        self.alphabet = alphabet
        self.seed = seed
        self._name = name
        random.seed(seed)

    def __call__(self, sequences: list[str]) -> MetricResult:
        """
        Compute mutational robustness score.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: Mean robustness score.
        """
        score = compute_mutational_robustness(
            sequences,
            predictor=self.predictor,
            n_mutations=self.n_mutations,
            alphabet=self.alphabet,
            seed=self.seed,
        )
        return MetricResult(score)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return "maximize"


def compute_mutational_robustness(
    sequences: list[str],
    *,
    predictor: Callable[[list[str]], np.ndarray],
    n_mutations: int = 3,
    alphabet: str = "ACDEFGHIKLMNPQRSTVWY",
    seed: int = 0,
) -> float:
    """
    Compute mutational robustness.

    For each sequence, introduce random point mutations and compare predictor outputs
    for original and mutated sequences.

    Args:
        sequences: Sequences to evaluate.
        predictor: Function returning numeric scores for sequences.
        n_mutations: Number of point mutations per sequence.
        alphabet: Allowed set of mutation characters.
        seed: Random seed for reproducibility.

    Returns:
        Mean robustness value in [-1, 1].
    """
    if not sequences:
        return np.nan

    rng = random.Random(seed)

    def mutate(seq: str) -> str:
        seq = list(seq)
        for _ in range(n_mutations):
            i = rng.randrange(len(seq))
            seq[i] = rng.choice(alphabet)
        return "".join(seq)

    # Compute predictor outputs
    orig_scores = np.array(predictor(sequences), dtype=float)
    mutated = [mutate(s) for s in sequences]
    mut_scores = np.array(predictor(mutated), dtype=float)

    var_orig = np.var(orig_scores)
    var_mut = np.var(mut_scores)

    robustness = 1 - var_mut / (var_orig + 1e-8)
    return float(np.clip(robustness, -1.0, 1.0))
