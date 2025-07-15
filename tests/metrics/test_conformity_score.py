import unittest
import random
from pepme.models.properties import Gravy
from pepme.metrics.conformity_score import ConformityScore


def generate_sequences_from_aas(aa_list, n_seqs, l=30):
    rng = random.Random(0)
    return ["".join(rng.choices(aa_list, k=l)) for _ in range(n_seqs)]


class TestConformityScore(unittest.TestCase):
    def test_conformity_score_match(self):
        neg_kd = set(["D", "E", "H", "R", "N"])  # AAs with low Kyte-Doolittle index
        sequences = generate_sequences_from_aas(list(neg_kd), 1100)
        reference = sequences[:1000]
        test = sequences[1000:]

        metric = ConformityScore(reference=reference, descriptors=[Gravy()])

        result = metric(test)
        self.assertAlmostEqual(result.value, 0.5, delta=0.05)

    def test_conformity_score_mismatch(self):
        neg_kd = set(["D", "E", "H", "R", "N"])  # AAs with low Kyte-Doolittle index
        pos_kd = set(["I", "L", "V", "R"])  # AAs with high Kyte-Doolittle index
        reference = generate_sequences_from_aas(list(neg_kd), 1000)
        test = generate_sequences_from_aas(list(pos_kd), 100)

        metric = ConformityScore(reference=reference, descriptors=[Gravy()])

        result = metric(test)
        self.assertAlmostEqual(result.value, 0.0, delta=0.05)


if __name__ == "__main__":
    unittest.main()
