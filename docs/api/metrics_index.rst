Metrics
#######

seqme contains three types of metrics. Those operating in sequence-, embedding-, and property-space.

Sequence-based
--------------

.. autosummary::
    :toctree:
    :nosignatures:

    seqme.metrics.Diversity
    seqme.metrics.Uniqueness
    seqme.metrics.Novelty
    seqme.metrics.KmerJaccardSimilarity

Embedding-based
---------------

.. autosummary::
    :toctree:
    :nosignatures:

    seqme.metrics.FBD
    seqme.metrics.MMD
    seqme.metrics.Precision
    seqme.metrics.Recall
    seqme.metrics.AuthPct

Property-based
--------------

.. autosummary::
    :toctree:
    :nosignatures:

    seqme.metrics.ID
    seqme.metrics.Threshold
    seqme.metrics.HitRate
    seqme.metrics.HV
    seqme.metrics.ConformityScore
    seqme.metrics.KLDivergence


Miscellaneous
-------------
.. autosummary::
    :toctree:
    :nosignatures:

    seqme.metrics.Fold
    seqme.metrics.Count
