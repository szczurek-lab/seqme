import pytest

from seqme.models import ESMFold

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def esm_fold():
    return ESMFold(batch_size=32, device="cpu")


def test_esm_fold_shape_and_means(esm_fold):
    sequences = [
        "RVKRVWPLVIRTVIAGYNLYRAIKKK",
        "RKRIHIGPGRAFYTT",
    ]
    embeddings = esm_fold(sequences)

    assert len(embeddings) == 2

    for i, sequence in enumerate(sequences):
        assert embeddings[i].shape == (len(sequence), 3)
