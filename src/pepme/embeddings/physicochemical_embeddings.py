from typing import Callable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pepme.properties.physicochemical_properties as php


class PhysicochemicalEmbedding:
    """
    Embedding model for peptide sequences based on selected physicochemical properties.

    Parameters
    ----------
    properties : list[str]
        List of property names to use. Options:
        ['charge', 'hydrophobicity', 'isoelectric_point', 'mol_weight',
         'instability_index', 'gravy', 'aliphatic_index', 'boman_index']
    scaling : str
        'none', 'minmax', or 'standard'
    ph : float
        pH used for charge calculation.
    hydro_scale : str
        Hydrophobicity scale for Peptide class.
    """

    _all_property_builders: dict[str, Callable[[float, str], Callable[[list[str]], np.ndarray]]] = {
        "charge": lambda ph, hs: php.Charge(ph),
        "hydrophobicity": lambda ph, hs: php.Hydrophobicity(hs),
        "isoelectric_point": lambda ph, hs: php.IsoelectricPoint(),
        "mol_weight": lambda ph, hs: php.MolecularWeight(),
        "instability_index": lambda ph, hs: php.InstabilityIndex(),
        "gravy": lambda ph, hs: php.Gravy(),
        "aliphatic_index": lambda ph, hs: php.AliphaticIndex(),
        "boman_index": lambda ph, hs: php.BomanIndex(),
    }

    def __init__(
        self,
        properties: list[str],
        scaling: str = "standard",
        ph: float = 7.0,
        hydro_scale: str = "Aboderin",
    ):
        self.scaling = scaling
        self.property_names = properties
        self.funcs: dict[str, Callable[[list[str]], np.ndarray]] = {
            name: self._all_property_builders[name](ph, hydro_scale)
            for name in properties
        }

    def __call__(self, sequences: list[str]) -> np.ndarray:
        """Returns np.ndarray of shape (n_sequences, n_features)."""
        df = self._raw_dataframe(sequences)

        if self.scaling == "none":
            return df.values
        elif self.scaling == "minmax":
            return self._minmax(df).values
        elif self.scaling == "standard":
            return self._standard(df).values
        else:
            raise ValueError("scaling must be 'none', 'minmax', or 'standard'")

    def _raw_dataframe(self, sequences: list[str]) -> pd.DataFrame:
        return pd.DataFrame(
            {name: func(sequences) for name, func in self.funcs.items()},
            index=np.arange(len(sequences)),
        )

    @staticmethod
    def _minmax(df: pd.DataFrame) -> pd.DataFrame:
        return (df - df.min()) / (df.max() - df.min())

    @staticmethod
    def _standard(df: pd.DataFrame) -> pd.DataFrame:
        return (df - df.mean()) / df.std(ddof=0)

    def plot_distributions(
        self, sequences: list[str], kind="hist", bins=30, figsize=(15, 8)
    ):
        """
        Plot distributions of selected properties.

        Parameters
        ----------
        sequences : list[str]
        kind : str
            One of {'hist', 'kde', 'box'}
        """
        df = self._raw_dataframe(sequences)

        if kind not in {"hist", "kde", "box"}:
            raise ValueError("kind must be 'hist', 'kde', or 'box'")

        n_cols = 3
        n_rows = -(-len(df.columns) // n_cols)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten()

        for i, col in enumerate(df.columns):
            ax = axes[i]
            if kind == "hist":
                ax.hist(df[col].dropna(), bins=bins, alpha=0.7)
            elif kind == "kde":
                df[col].dropna().plot(kind="kde", ax=ax)
            elif kind == "box":
                ax.boxplot(df[col].dropna(), vert=True)
                ax.set_xticklabels([col])
                continue
            ax.set_title(col)
            ax.grid(True)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        plt.show()
