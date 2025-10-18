Metrics
#######
``seqme`` provides a unified framework for evaluating generated across **three metric spaces** — sequence, embedding, and property — along with a few general-purpose utilities.


Sequence-based Metrics
----------------------
Metrics that operate directly on the raw sequence data.

.. autosummary::
    :toctree:
    :nosignatures:

    seqme.metrics.Diversity
    seqme.metrics.Uniqueness
    seqme.metrics.Novelty
    seqme.metrics.NGramJaccardSimilarity


Embedding-based Metrics
-----------------------
Metrics that compare or assess distributions in an embedding (vector) space.

.. autosummary::
    :toctree:
    :nosignatures:

    seqme.metrics.FBD
    seqme.metrics.MMD
    seqme.metrics.Precision
    seqme.metrics.Recall
    seqme.metrics.AuthPct
    seqme.metrics.FKEA


Property-based Metrics
----------------------
Metrics computed on derived physicochemical or predicted properties.

.. autosummary::
    :toctree:
    :nosignatures:

    seqme.metrics.ID
    seqme.metrics.Threshold
    seqme.metrics.HitRate
    seqme.metrics.Hypervolume
    seqme.metrics.ConformityScore
    seqme.metrics.KLDivergence


Miscellaneous
-------------
General or utility metrics that don't fit into the main categories.

.. autosummary::
    :toctree:
    :nosignatures:

    seqme.metrics.Fold
    seqme.metrics.Count
    seqme.metrics.Length
