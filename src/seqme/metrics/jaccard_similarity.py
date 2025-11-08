from typing import Literal

from seqme.core.base import Metric, MetricResult


class NGramJaccardSimilarity(Metric):
    r"""
    Average Jaccard similarity between each generated sequence and a reference corpus, based on n-grams of size ``n``, using \|A ∩ R\| / \|A ∪ R\|.

    You can choose to ``'minimize'`` (novelty) or ``'maximize'`` (overlap) via the ``objective`` parameter.
    """

    def __init__(
        self,
        reference: list[str],
        n: int,
        *,
        objective: Literal["minimize", "maximize"] = "minimize",
        name: str = "Jaccard-similarity",
    ):
        """Initialize the metric.

        Args:
            reference: list of strings to build the reference n-gram set.
            n: size of the n-grams.
            objective: ``"minimize"`` to reward novelty, ``"maximize"`` to reward overlap.
            name: Metric name.
        """
        if n < 1:
            raise ValueError("Expected n >= 1.")

        self.n = n
        self._objective = objective
        self.reference_ngrams = self._make_ngram_set(reference)
        self._name = name

    def _make_ngram_set(self, corpus: list[str]) -> set[str]:
        all_ngrams: set[str] = set()
        for seq in corpus:
            all_ngrams |= self._ngrams(seq)
        return all_ngrams

    def _ngrams(self, seq: str) -> set[str]:
        L = len(seq)
        if L < self.n:
            return set()
        return {seq[i : i + self.n] for i in range(L - self.n + 1)}

    def __call__(self, sequences: list[str]) -> MetricResult:
        """Compute the average Jaccard similarity between each generated sequence and a reference corpus, based on n-grams of size ``n``.

        Args:
            sequences: Sequences to evaluate.

        Returns:
            MetricResult: Jaccard similarity.
        """
        total = len(sequences)
        if total == 0:
            return MetricResult(0.0)

        sim_sum = 0.0
        R = self.reference_ngrams

        for seq in sequences:
            A = self._ngrams(seq)
            union = A | R
            if not union:
                # both A and R empty → define similarity = 0
                continue
            sim_sum += len(A & R) / len(union)

        score = sim_sum / total
        return MetricResult(score)

    @property
    def name(self) -> str:
        return self._name

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return self._objective
