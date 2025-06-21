from typing import List, Literal

import numpy as np
import peptides

from pepme.base import Metric, MetricResult


class Gravy(Metric):
    def __call__(self, sequences: List[str]) -> MetricResult:
        values = [
            peptides.Peptide(sequence).charge() / len(sequence)
            for sequence in sequences
        ]
        return MetricResult(np.mean(values), deviation=np.std(values))  # type: ignore

    @property
    def name(self) -> str:
        return "Gravy"

    @property
    def objective(self) -> Literal["minimize", "maximize", "ambiguous"]:
        return "ambiguous"
