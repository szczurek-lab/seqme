import unittest

import numpy as np

from pepme.embeddings.esm2 import ESM2


class TestFid(unittest.TestCase):
    def test_esm2(self):
        data = [
            "RVKRVWPLVIRTVIAGYNLYRAIKKK",
            "RKRIHIGPGRAFYTT",
            "DSHAKRHHGYKRKFHEKHHSHRGY",
            "ENREVPPGFTALIKTLRKCKII",
            "NLVSGLIEARKYLEQLHRKLKNCKV",
            "FLPKTLRKFFARIRGGRAAVLNALGKEEQIGRASNSGRKCARKKK",
        ]
        esm = ESM2(model_name="esm2_t6_8M_UR50D", batch_size=32, device="cpu")
        embeddings = esm(data)
        assert embeddings.shape == (6, 320)
        assert np.allclose(
            embeddings.mean(axis=-1),
            np.array(
                [
                    -0.01061969,
                    -0.01052918,
                    -0.01140676,
                    -0.00957893,
                    -0.00982053,
                    -0.0104174,
                ]
            ),
        )


if __name__ == "__main__":
    unittest.main()
