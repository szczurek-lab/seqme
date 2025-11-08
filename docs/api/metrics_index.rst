Metrics
#######
``seqme`` provides a unified framework for evaluating sequences across **three metric spaces** — sequence, embedding, and property — along with a few general-purpose utilities.


Sequence-based Metrics
----------------------
Metrics that operate directly on the raw sequences.

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
    seqme.metrics.KID
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
    seqme.metrics.Subset
    seqme.metrics.Count
    seqme.metrics.Length


.. |ok| image:: /_static/green-check.svg
   :alt: ✓
   :class: icon

.. |no| image:: /_static/gray-cross.svg
   :alt: ✗
   :class: icon


Supported sequence types
------------------------
**At-a-glance matrix of all metrics and supported sequence types.**

|ok| — supported, |no| — not supported

.. list-table::
   :header-rows: 1
   :widths: 36 10 10 10 10 10
   :align: center

   * - **Metrics**
     - **Protein**
     - **Peptide**
     - **RNA**
     - **DNA**
     - **Small Molecule**
   * - :py:class:`seqme.metrics.Diversity`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |no|
   * - :py:class:`seqme.metrics.Uniqueness`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.Novelty`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.NGramJaccardSimilarity`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |no|
   * - :py:class:`seqme.metrics.FBD`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.MMD`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.KID`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.Precision`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.Recall`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.AuthPct`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.FKEA`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.ID`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.Threshold`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.HitRate`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.Hypervolume`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.ConformityScore`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.KLDivergence`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.Fold`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.Subset`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.Count`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.metrics.Length`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |no|
