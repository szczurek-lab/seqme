import unittest

import pandas as pd

from pepme.core import compute_metrics
from pepme.metrics.novelty import Novelty


class TestComputeMetrics(unittest.TestCase):
    def test_compute_metrics(self):
        sequences = {"my_model": ["KKW", "RRR", "RRR"]}
        metrics = [Novelty(reference=["KKW"])]

        df = compute_metrics(sequences, metrics)

        self.assertEqual(df.shape, (1, 2))
        self.assertEqual(df.attrs["objective"], {"Novelty": "maximize"})
        self.assertEqual(df.index.tolist(), ["my_model"])
        self.assertEqual(
            df.columns.tolist(), [("Novelty", "value"), ("Novelty", "deviation")]
        )
        self.assertAlmostEqual(df.at["my_model", ("Novelty", "value")], 2.0 / 3.0)
        self.assertTrue(pd.isna(df.at["my_model", ("Novelty", "deviation")]))

    def test_empty_metrics_list(self):
        sequences = {"random": ["MKQW", "RKSPL"]}
        metrics = []

        with self.assertRaisesRegex(ValueError, "^No metrics provided$"):
            compute_metrics(sequences, metrics)

    def test_duplicate_metric_names(self):
        sequences = {"random": ["MKQW", "RKSPL"]}

        metrics = [
            Novelty(reference=["KKW", "RKSPL"]),
            Novelty(reference=["RASD", "KKKQ", "LPTUY"]),
        ]

        with self.assertRaisesRegex(
            ValueError,
            "^Metrics must have unique names. Found duplicates: Novelty, Novelty$",
        ):
            compute_metrics(sequences, metrics)


if __name__ == "__main__":
    unittest.main()
