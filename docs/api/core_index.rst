Core
####
Essential functionality for evaluating, manipulating, and visualizing metrics.

This module provides a high-level interface for working with metric tables, managing results, and performing common data manipulation and visualization tasks.


Base Components
---------------
Core objects and classes that define the ``seqme`` evaluation and metric API.

.. autosummary::
    :toctree:
    :nosignatures:

    seqme.evaluate
    seqme.Cache
    seqme.Metric
    seqme.MetricResult


Data Manipulation
-----------------
Utility functions to combine, filter, and reorganize metric results.

.. autosummary::
    :toctree:
    :nosignatures:

    seqme.combine
    seqme.rank
    seqme.top_k
    seqme.sort
    seqme.rename


Visualization
-------------
Functions for visual exploration and presentation of metrics.

.. autosummary::
    :toctree:
    :nosignatures:

    seqme.show
    seqme.plot_bar
    seqme.plot_scatter
    seqme.plot_parallel
    seqme.plot_line
    seqme.to_latex

Input / Output
--------------
Functions for serializing and loading data and metric results.

.. autosummary::
    :toctree:
    :nosignatures:

    seqme.to_pickle
    seqme.read_pickle
    seqme.to_fasta
    seqme.read_fasta
