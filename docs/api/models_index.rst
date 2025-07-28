Models
######
Models mapping sequences to either embedding- or property-space.

Models
------
.. autosummary::
    :toctree:
    :nosignatures:

    seqme.models.Esm2
    seqme.models.KmerFrequencyEmbedding

    seqme.models.AliphaticIndex
    seqme.models.Aromaticity
    seqme.models.BomanIndex
    seqme.models.Charge
    seqme.models.Gravy
    seqme.models.Hydrophobicity
    seqme.models.HydrophobicMoment
    seqme.models.InstabilityIndex
    seqme.models.IsoelectricPoint
    seqme.models.MolecularWeight

Miscellaneous
-------------
.. autosummary::
    :toctree:
    :nosignatures:

    seqme.models.ThirdPartyModel
    seqme.models.Ensemble
    seqme.models.Concatenate
    seqme.models.normalizers.MinMaxNorm


Diagnostics
-----------
.. autosummary::
    :toctree:
    :nosignatures:

    seqme.models.diagnostics.feature_alignment_score
    seqme.models.diagnostics.spearman_correlation_coefficient
    seqme.models.diagnostics.plot_feature_alignment_score