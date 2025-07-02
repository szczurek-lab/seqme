import numpy as np
import pandas as pd
from peptides import Peptide
import matplotlib.pyplot as plt
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class Gravy:
    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([ProteinAnalysis(seq).gravy() for seq in sequences])

class Charge:
    """Net charge at a given pH (default 7.0)."""
    def __init__(self, ph: float = 7.0):
        self.ph = ph

    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([ProteinAnalysis(seq).charge_at_pH(self.ph) for seq in sequences])

class Hydrophobicity:
    """Average hydrophobicity on a chosen scale (default: Aboderin)."""
    def __init__(self, scale: str = "Aboderin"):
        self.scale = scale

    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([Peptide(seq).hydrophobicity(scale=self.scale) for seq in sequences])


class IsoelectricPoint:
    """Theoretical pI."""
    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([ProteinAnalysis(seq).isoelectric_point() for seq in sequences])


class MolecularWeight:
    """Average molecular mass in Daltons."""
    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([ProteinAnalysis(seq).molecular_weight() for seq in sequences])


class InstabilityIndex:
    """Guruprasad instability index (values > 40 ≈ unstable)."""
    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([ProteinAnalysis(seq).instability_index() for seq in sequences])


class AliphaticIndex:
    """Relative volume occupied by aliphatic side chains (A + V + I + L)."""
    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([Peptide(seq).aliphatic_index() for seq in sequences])


class BomanIndex:
    """Predicted protein–binding potential (high |value| ⇒ strong binding)."""
    def __call__(self, sequences: list[str]) -> np.ndarray:
        return np.array([Peptide(seq).boman() for seq in sequences])

class PhysicochemicalPropertyAggregator:
    """
    Compute eight physicochemical properties for an iterable of peptide sequences
    and optionally scale them.

    Scaling modes:
        - 'none'      : raw numeric values
        - 'minmax'    : (x - min) / (max - min)  →  range [0, 1]
        - 'standard'  : (x - mean) / std         →  mean 0, std 1
    """

    def __init__(self, ph: float = 7.0, hydro_scale: str = "Aboderin"):
        # Instantiate the property calculators -------------------------------
        self._props = {
            "charge"             : Charge(ph=ph),
            "hydrophobicity"     : Hydrophobicity(scale=hydro_scale),
            "isoelectric_point"  : IsoelectricPoint(),
            "mol_weight"         : MolecularWeight(),
            "instability_index"  : InstabilityIndex(),
            "gravy"              : Gravy(),
            "aliphatic_index"    : AliphaticIndex(),
            "boman_index"        : BomanIndex(),
        }

    # --------------------------------------------------------------------- #
    #  Public helpers                                                        #
    # --------------------------------------------------------------------- #
    def compute(self, sequences: list[str], scaling: str = "standard") -> pd.DataFrame:
        """
        Return a DataFrame with one row per sequence and one column per property.

        Parameters
        ----------
        sequences : list[str]
            Peptide primary structures (e.g. "GIGAVLKVLTT").
        scaling : {'none', 'minmax', 'standard'}
            Desired scaling of numeric columns.

        Returns
        -------
        pd.DataFrame
        """
        df_raw = self._raw_dataframe(sequences)

        if scaling == "none":
            return df_raw

        if scaling == "minmax":
            return self._minmax(df_raw)

        if scaling == "standard":
            return self._standard(df_raw)

        raise ValueError("scaling must be 'none', 'minmax', or 'standard'")
    
    def plot_property_distributions(self, sequences, kind="hist", bins=30, figsize=(15, 8)):
        """
        Plot the distribution of physicochemical properties in a DataFrame.

        Parameters:
            sequences (list): List of peptides to visualize the distributions
            kind (str): Type of plot. One of {'hist', 'box', 'kde'}.
            bins (int): Number of bins for histograms.
            figsize (tuple): Size of the full figure.
        """
        if kind not in {"hist", "box", "kde"}:
            raise ValueError("kind must be one of {'hist', 'box', 'kde'}")
        
        df = self.compute(sequences)

        n_cols = 3
        n_rows = -(-len(df.columns) // n_cols)  # ceiling division
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i, col in enumerate(df.columns):
            ax = axes[i]
            if kind == "hist":
                ax.hist(df[col].dropna(), bins=12, alpha=0.7)
            elif kind == "kde":
                df[col].dropna().plot(kind="kde", ax=ax)
            elif kind == "box":
                ax.boxplot(df[col].dropna(), vert=True)
                ax.set_xticklabels([col])
                continue
            ax.set_title(col)
            ax.grid(True)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        plt.show()

    def _raw_dataframe(self, sequences: list[str]) -> pd.DataFrame:
        """Compute *un-scaled* property values."""
        data = {name: func(sequences) for name, func in self._props.items()}
        return pd.DataFrame(data, index=np.arange(len(sequences)))

    @staticmethod
    def _minmax(df: pd.DataFrame) -> pd.DataFrame:
        """Min–max scale numeric columns to [0, 1]."""
        return (df - df.min()) / (df.max() - df.min())

    @staticmethod
    def _standard(df: pd.DataFrame) -> pd.DataFrame:
        """Z-score standardisation (mean 0, std 1)."""
        return (df - df.mean()) / df.std(ddof=0)
