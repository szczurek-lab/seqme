from typing import Literal

from seqme.core import Metric, MetricResult


class KmerJaccardSimilarity(Metric):
    r"""
    Average Jaccard similarity between each generated sequence
    and a reference corpus, based on n-grams of size `n`, using
    \|A ∩ R\| / \|A ∪ R\|. You can choose to 'minimize' (novelty)
    or 'maximize' (overlap) via the `objective` parameter.
    """

    def __init__(
        self,
        reference: list[str],
        n: int,
        *,
        objective: Literal["minimize", "maximize"] = "minimize",
        reference_name: str | None = None,
    ):
        """
        Args:
            reference: list of strings to build the reference n-gram set.
            n: size of the n-grams.
            objective: "minimize" to reward novelty, "maximize" to reward overlap.
            reference_name: optional label; appended to the metric name.
        """
        self.n = n
        self._objective = objective
        self.data_name = reference_name
        self.reference_ngrams = self._make_ngram_set(reference)

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
        """
        Computes the average Jaccard similarity between each generated sequence
        and a reference corpus, based on n-grams of size `n`.

        Args:
            sequences: A list of generated sequences to evaluate.

        Returns:
            MetricResult containing the Jaccard similarity.
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
        base = f"Jaccard-{self.n}"
        return base if self.data_name is None else f"{base} ({self.data_name})"

    @property
    def objective(self) -> Literal["minimize", "maximize"]:
        return self._objective
