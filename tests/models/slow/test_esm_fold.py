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
    folds = esm_fold.fold(sequences, convention="atom37", compute_ptm=False, output_pdb=True, return_type="dict")
    embeddings = folds["positions"]

    assert len(embeddings) == 2

    for i, sequence in enumerate(sequences):
        assert embeddings[i].shape == (len(sequence), 37, 3)

    mean_plddts = [plddt.mean().item() for plddt in folds["plddt"]]
    assert mean_plddts == pytest.approx([0.882698, 0.822512])
