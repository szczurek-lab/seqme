import pytest

from seqme.models import EsmFold

pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def esm_fold():
    return EsmFold(batch_size=32, device="cpu")


def test_esm_fold_shape_and_means(esm_fold):
    data = [
        "RVKRVWPLVIRTVIAGYNLYRAIKKK",
        "RKRIHIGPGRAFYTT",
        "DSHAKRHHGYKRKFHEKHHSHRGY",
    ]
    embeddings = esm_fold.fold(data, convention="atom14")

    assert len(embeddings) == 3

    for i, sequence in enumerate(data):
        assert embeddings[i].shape == (len(sequence), 14, 3)
