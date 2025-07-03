import numpy as np

from pepme.embeddings.huggingface_embeddings import compute_huggingface_model_embeddings


def test_esm2():
    data = [
        "RVKRVWPLVIRTVIAGYNLYRAIKKK",
        "RKRIHIGPGRAFYTT",
        "DSHAKRHHGYKRKFHEKHHSHRGY",
        "ENREVPPGFTALIKTLRKCKII",
        "NLVSGLIEARKYLEQLHRKLKNCKV",
        "FLPKTLRKFFARIRGGRAAVLNALGKEEQIGRASNSGRKCARKKK",
    ]
    embeddings = compute_huggingface_model_embeddings(data, opt="esm2_t6_8M_UR50D")
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
