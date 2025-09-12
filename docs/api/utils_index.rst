Utils
#####
Utility functions for models and metrics.
Plotting functionality for models mapping sequences to either embedding- or property-space. Diagnostics to evaluate embedding models alignment with the feature(s) of interest.

Plots
-----
.. autosummary::
    :toctree:
    :nosignatures:

    seqme.utils.plot_embeddings
    seqme.utils.plot_embedding_with_value
    seqme.utils.plot_hist
    seqme.utils.plot_kde
    seqme.utils.plot_violin

Projections
-----------
.. autosummary::
    :toctree:
    :nosignatures:

    seqme.utils.pca
    seqme.utils.tsne
    seqme.utils.umap


Diagnostics
-----------
.. autosummary::
    :toctree:
    :nosignatures:

    seqme.utils.feature_alignment_score
    seqme.utils.spearman_correlation_coefficient
    seqme.utils.plot_feature_alignment_score


Sequence manipulation
---------------------
.. autosummary::
    :toctree:
    :nosignatures:

    seqme.utils.shuffle_sequences
    seqme.utils.random_subset


IO
---
.. autosummary::
    :toctree:
    :nosignatures:

    seqme.utils.read_fasta_file
    seqme.utils.write_to_fasta_file
    