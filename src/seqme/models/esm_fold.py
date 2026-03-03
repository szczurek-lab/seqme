from collections import defaultdict
from typing import Any, Literal

import numpy as np
import torch
from tqdm import tqdm

from .exceptions import OptionalDependencyError


class ESMFold:
    """
    ESMFold protein language model.

    The model predicts the 3D-structure (fold) of a protein sequence.

    Installation: ``pip install "seqme[esmfold]"``

    Reference:
        Lin et al., "Language models of protein sequences at the scale of evolution enable accurate structure prediction"
        (https://www.biorxiv.org/content/10.1101/2022.07.20.500902v3)
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int = 256,
        cache_dir: str | None = None,
        verbose: bool = False,
    ):
        """
        Initialize the ESMFold model.

        Args:
            device: Device to run inference on, e.g., ``"cuda"`` or ``"cpu"``.
            batch_size: Number of sequences to process per batch.
            cache_dir: Directory to cache the model.
            verbose: Whether to display a progress bar.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose

        try:
            from transformers import AutoTokenizer, EsmForProteinFolding
        except ModuleNotFoundError:
            raise OptionalDependencyError("esmfold") from None

        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1", cache_dir=cache_dir)
        self.model = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", cache_dir=cache_dir)

        self.model.to(device)
        self.model.eval()

    def __call__(self, sequences: list[str]) -> list[np.ndarray]:
        fold = self.fold(sequences, convention="ca", compute_ptm=False, output_pdb=False, return_type="dict")
        return fold["positions"]  # type: ignore

    @torch.inference_mode()
    def fold(
        self,
        sequences: list[str],
        *,
        convention: Literal["atom14", "atom37", "ca"] = "ca",
        compute_ptm: bool = True,
        output_pdb: bool = True,
        return_type: Literal["dict", "list"] = "list",
    ) -> dict[str, list] | list[dict]:
        """
        Predict protein sequences TM-score, pLDDT and 3D-structure, i.e., atom coordinates.

        Args:
            sequences: Protein sequences which 3D-structure is predicted.
            convention: The position/coordinates encoding of the atoms.

                - ``'atom14'``:
                    Atom position/coordinates follow this mapping:

                        residue_atoms: dict[str, list[str]] = {
                            "ALA": ["C", "CA", "CB", "N", "O"],
                            "ARG": ["C", "CA", "CB", "CG", "CD", "CZ", "N", "NE", "O", "NH1", "NH2"],
                            "ASP": ["C", "CA", "CB", "CG", "N", "O", "OD1", "OD2"],
                            "ASN": ["C", "CA", "CB", "CG", "N", "ND2", "O", "OD1"],
                            "CYS": ["C", "CA", "CB", "N", "O", "SG"],
                            "GLU": ["C", "CA", "CB", "CG", "CD", "N", "O", "OE1", "OE2"],
                            "GLN": ["C", "CA", "CB", "CG", "CD", "N", "NE2", "O", "OE1"],
                            "GLY": ["C", "CA", "N", "O"],
                            "HIS": ["C", "CA", "CB", "CG", "CD2", "CE1", "N", "ND1", "NE2", "O"],
                            "ILE": ["C", "CA", "CB", "CG1", "CG2", "CD1", "N", "O"],
                            "LEU": ["C", "CA", "CB", "CG", "CD1", "CD2", "N", "O"],
                            "LYS": ["C", "CA", "CB", "CG", "CD", "CE", "N", "NZ", "O"],
                            "MET": ["C", "CA", "CB", "CG", "CE", "N", "O", "SD"],
                            "PHE": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O"],
                            "PRO": ["C", "CA", "CB", "CG", "CD", "N", "O"],
                            "SER": ["C", "CA", "CB", "N", "O", "OG"],
                            "THR": ["C", "CA", "CB", "CG2", "N", "O", "OG1"],
                            "TRP": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2", "N", "NE1", "O"],
                            "TYR": ["C", "CA", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "N", "O", "OH"],
                            "VAL": ["C", "CA", "CB", "CG1", "CG2", "N", "O"],
                        }

                        Mapping is from here: https://github.com/huggingface/transformers/blob/99b0995138c17ef953959c70f35cb2bdc41111a2/src/transformers/models/esm/openfold_utils/residue_constants.py#L335

                        Amino-acids are defined by at most 14 atoms (excluding hydrogens). The shape of a sequences fold is [sequence_length, 14, 3],
                        where the dimension with 14 elements, corresponds to an amino acids atom positions, and the dimension with 3 elements corresponds to "xyz"-coordinates. If an amino-acid has fewer than 14 atoms, then those positions should be discarded / ignored as they are unused.

                - ``atom37``:
                    Atom position/coordinates follow this mapping:

                        atom_types: list[str] = [
                            "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
                            "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1","CE2", "CE3",
                            "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2",
                            "CZ3", "NZ", "OXT",
                        ]

                        Mapping is from here: https://github.com/huggingface/transformers/blob/99b0995138c17ef953959c70f35cb2bdc41111a2/src/transformers/models/esm/openfold_utils/residue_constants.py#L500

                - ``'ca'``:
                    Carbon alphas (CA) position.


            compute_ptm: If ``True``, computes the ptm score (structure confidence score) but reduces the batch size to 1 in order to do so.
            output_pdb: Whether to return the 3D-structure encoded as a PDB for each sequence.
            return_type: If ``"list"``, return list of dict else if ``"dict"`` return dict of lists.

        Returns:
            A dict with
                "position": Numpy arrays of shape:

                    - "atom14": sequence_length x 14 x 3
                    - "atom37": sequence_length x 37 x 3
                    - "ca": sequence_length x 3

                "plddt": Numpy arrays of shape: sequence_length (pLDDT for carbon alpha atom)
                "ptm": predicted TM-scores if `compute_ptm` is true.
                "pdb": PDBs if `output_pdb` is true.
        """
        batch_size = 1 if compute_ptm else self.batch_size

        folds: dict[str, list] = defaultdict(list)

        for start in tqdm(range(0, len(sequences), batch_size), disable=not self.verbose):
            batch = sequences[start : start + batch_size]

            tokens = self.tokenizer(
                batch,
                return_tensors="pt",
                add_special_tokens=False,
                padding=True,
                truncation=False,
            )

            tokens = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in tokens.items()}

            outputs = self.model(**tokens)

            atom14 = outputs.positions[-1]
            plddt = outputs.plddt

            lengths = [len(seq) for seq in batch]
            B = atom14.shape[0]

            if convention == "ca":
                positions = [atom14[i, :L, 1].cpu().numpy() for i, L in enumerate(lengths)]
            elif convention == "atom14":
                positions = [atom14[i, :L].cpu().numpy() for i, L in enumerate(lengths)]
            elif convention == "atom37":
                from transformers.models.esm.openfold_utils.feats import atom14_to_atom37

                atom37 = atom14_to_atom37(atom14, outputs)
                positions = [atom37[i, :L].cpu().numpy() for i, L in enumerate(lengths)]
            else:
                raise ValueError(f"Unsupported convention: '{convention}'.")

            folds["positions"].extend(positions)
            folds["plddt"].extend(plddt[i, :L, 1].cpu().numpy() for i, L in enumerate(lengths))

            if output_pdb:
                pdbs = _convert_outputs_to_pdb(outputs)
                folds["pdb"].extend(pdbs)

            if compute_ptm:
                ptm_val = outputs.ptm.item()
                folds["ptm"].extend([ptm_val] * B)

        if return_type == "dict":
            return folds

        if return_type == "list":
            return _dict_to_list(folds)

        raise ValueError(f"Invalid return_type: '{return_type}'.")


def _dict_to_list(_dict: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(_dict.keys())
    return [dict(zip(keys, vals, strict=True)) for vals in zip(*_dict.values(), strict=True)]


# Adapted from: https://github.com/huggingface/notebooks/blob/main/examples/protein_folding.ipynb
def _convert_outputs_to_pdb(outputs: dict[str, Any]) -> list[str]:
    from transformers.models.esm.openfold_utils.feats import atom14_to_atom37
    from transformers.models.esm.openfold_utils.protein import Protein, to_pdb

    atom_positions = atom14_to_atom37(outputs["positions"][-1], outputs).cpu().numpy()
    outputs = {k: v.cpu().numpy() for k, v in outputs.items()}

    atom_masks = outputs["atom37_atom_exists"]
    aatypes = outputs["aatype"]
    res_ids = outputs["residue_index"]
    plddts = outputs["plddt"]

    pdbs = []
    for i in range(aatypes.shape[0]):
        prot = Protein(
            aatype=aatypes[i],
            atom_positions=atom_positions[i],
            atom_mask=atom_masks[i],
            residue_index=res_ids[i] + 1,
            b_factors=plddts[i],
            chain_index=outputs["chain_index"][i] if "chain_index" in outputs else None,
        )

        pdb = to_pdb(prot)
        pdbs.append(pdb)

    return pdbs
