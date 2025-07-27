Models
######
Models mapping sequences to either embedding- or property-space.

Models
------
.. autosummary::
    :toctree:
    :nosignatures:

    pepme.models.Esm2
    pepme.models.KmerFrequencyEmbedding

    pepme.models.AliphaticIndex
    pepme.models.Aromaticity
    pepme.models.BomanIndex
    pepme.models.Charge
    pepme.models.Gravy
    pepme.models.Hydrophobicity
    pepme.models.HydrophobicMoment
    pepme.models.InstabilityIndex
    pepme.models.IsoelectricPoint
    pepme.models.MolecularWeight

Miscellaneous
-------------
.. autosummary::
    :toctree:
    :nosignatures:

    pepme.models.ThirdPartyModel
    pepme.models.Ensemble
    pepme.models.Concatenate
    pepme.models.normalizers.MinMaxNorm


Diagnostics
-----------
.. autosummary::
    :toctree:
    :nosignatures:

    pepme.models.diagnostics.feature_alignment_score
    pepme.models.diagnostics.spearman_correlation_coefficient
    pepme.models.diagnostics.plot_feature_alignment_score