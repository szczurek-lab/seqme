from typing import List, Literal

import numpy as np
import peptides

from pepme.core import Metric, MetricResult


class Gravy(Metric):
    def __call__(self, sequences: List[str]) -> MetricResult:
        values = [
            peptides.Peptide(sequence).hydrophobicity() / len(sequence)
            for sequence in sequences
        ]
        return MetricResult(np.mean(values).item(), deviation=np.std(values).item())

    @property
    def name(self) -> str:
        return "Gravy"

    @property
    def objective(self) -> Literal["minimize", "maximize", "ambiguous"]:
        return "ambiguous"
