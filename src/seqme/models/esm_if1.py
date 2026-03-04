import numpy as np
import torch
from tqdm import tqdm

from .exceptions import OptionalDependencyError


class ESMIF1:
    """
    Wrapper for the ESM inverse folding (ESM-IF1) model.

    Installation: ``pip install "seqme[esmif1]"``

    Note:
        If you have an issue with installing ESM-IF1 on a machine with cuda support due to torch-scatter,
        try running ``pip install torch-scatter --no-build-isolation`` first.

    Warning:
        Experimental. May change in the future or get removed.

    Examples:
        >>> sequences = ["MKRM", "KKRPR"]
        >>> folder = sm.models.ESMFold()  # Folding model
        >>> folds = folder.fold(sequences, convention="atom37", compute_ptm=False, return_type="dict")
        >>> atom_indices = [0, 1, 2]  # atoms: N, CA, C
        >>> coords = [seq_pos[:, atom_indices, :] for seq_pos in folds["positions"]]
        >>> inv_folder = sm.models.ESMIF1()  # Inverse folding model
        >>> inv_folder.compute_perplexity(coords, sequences)  # scPerplexity

    Reference:
        Hsu et al., "Learning inverse folding from millions of predicted structures"
        (https://www.biorxiv.org/content/10.1101/2022.04.10.487779v2)
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int = 256,
        verbose: bool = False,
    ):
        """
        Initialize the model.

        Args:
            device: Device to run inference on, e.g., ``"cuda"`` or ``"cpu"``.
            batch_size: Number of sequences to process per batch.
            verbose: Whether to display a progress bar.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.batch_size = batch_size
        self.device = device
        self.verbose = verbose

        try:
            _patch_import()
            import esm
        except ModuleNotFoundError:
            raise OptionalDependencyError("esmif1") from None

        self.model, self.alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

        self.model.to(device)
        self.model.eval()

    @torch.inference_mode()
    def compute_perplexity(
        self,
        coordinates: list[np.ndarray],
        sequences: list[str],
    ) -> np.ndarray:
        """
        Compute perplexity after inverse folding the backbones (coordinates) and comparing against the target sequences.

        Args:
            coordinates: List of amino acid coordinates. Each entry: len(sequence) x 3 x 3 for N, CA, C atoms.
            sequences: Amino acid sequences associated with the coordinates (backbone).

        Returns:
            np.ndarray: Perplexity scores.
        """
        perplexities = []
        for i in tqdm(range(0, len(sequences), self.batch_size), disable=not self.verbose):
            batch_coordinates = coordinates[i : i + self.batch_size]
            batch_sequences = sequences[i : i + self.batch_size]

            batch_perplexities = _compute_perplexity(
                model=self.model,
                alphabet=self.alphabet,
                coords=batch_coordinates,
                sequences=batch_sequences,
                device=self.device,
            )

            perplexities.append(batch_perplexities)

        return np.concatenate(perplexities)


# @NOTE: esm inverse folding model (esm-fair) imports 'filter_backbone' globally from an older version of biotite (v0.41)
# which has been renamed in v1.0.0. However we don't use that function, so define a placeholder function so we can import
# the esm inverse folding model.
def _patch_import():
    import biotite.structure as _bs

    if hasattr(_bs, "filter_backbone"):
        return

    def filter_backbone(*args, **kwargs):
        raise NotImplementedError("filter_backbone was removed from biotite; dependency still imports it.")

    _bs.filter_backbone = filter_backbone


# Adapted from: https://github.com/facebookresearch/esm/blob/main/esm/inverse_folding/util.py#L125
def _tokenize(alphabet, coords: list[np.ndarray], sequences: list[str], device: str):
    from esm.inverse_folding.util import CoordBatchConverter

    batch_converter = CoordBatchConverter(alphabet)
    coords, confidence, _, tokens, padding_mask = batch_converter.from_lists(
        coords_list=coords, seq_list=sequences, device=device
    )
    prev_output_tokens = tokens[:, :-1]
    target = tokens[:, 1:]
    target_mask = target != alphabet.padding_idx

    return coords, padding_mask, confidence, prev_output_tokens, target, target_mask


def _logits_to_perplexity(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_nll = torch.nn.functional.cross_entropy(logits, targets, reduction="none")
    nll = (token_nll * mask).sum(dim=1) / mask.sum(dim=1)
    return torch.exp(nll)


def _compute_perplexity(
    model,
    alphabet,
    coords: list[np.ndarray],
    sequences: list[str],
    device: str,
) -> np.ndarray:
    coords, padding_mask, confidence, prev_output_tokens, targets, target_mask = _tokenize(
        alphabet=alphabet, coords=coords, sequences=sequences, device=device
    )
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
    perplexities = _logits_to_perplexity(logits, targets, target_mask).cpu()

    return perplexities.cpu().numpy()
