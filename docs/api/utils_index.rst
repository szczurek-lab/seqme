Utils
#####
Utility functions for models and metrics.
Plotting functionality for models mapping sequences to either embedding- or property-space. Diagnostics to evaluate embedding models alignment with the feature(s) of interest.


Projections
-----------
.. autosummary::
    :toctree:
    :nosignatures:

    seqme.utils.pca
    seqme.utils.tsne
    seqme.utils.umap
    seqme.utils.plot_embeddings


Diagnostics
-----------
.. autosummary::
    :toctree:
    :nosignatures:

    seqme.utils.spearman_alignment_score
    seqme.utils.knn_alignment_score
    seqme.utils.plot_knn_alignment_score


    