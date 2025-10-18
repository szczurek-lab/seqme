Models
======

Mapping biological sequences to numerical representations.

Overview
--------
The ``seqme`` library provides a suite of models that convert biomolecular sequences into numerical
representations. These models fall into two main categories:

* **Embedding models** — map sequences to high-dimensional vector spaces.
* **Property models** — compute interpretable scalar or descriptor values such as charge, weight, or hydrophobicity.

Models
------
.. autosummary::
    :toctree:
    :nosignatures:

    seqme.models.Esm2
    seqme.models.EsmFold
    seqme.models.ProstT5
    seqme.models.RNA_FM
    seqme.models.GenaLM
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
    seqme.models.ProteinWeight

Miscellaneous
-------------
These models perform special-purpose operations or combine multiple models.

.. autosummary::
    :toctree:
    :nosignatures:

    seqme.models.ThirdPartyModel
    seqme.models.PCA
    seqme.models.Ensemble


.. |ok| image:: /_static/green-check.svg
   :alt: ✓
   :class: icon

.. |no| image:: /_static/gray-cross.svg
   :alt: ✗
   :class: icon


Supported sequence types
------------------------
**At-a-glance matrix of all models and supported sequence types.**

|ok| — supported, |no| — not supported

.. list-table::
   :header-rows: 1
   :widths: 36 10 10 10 10 10
   :align: center

   * - **Model**
     - **Protein**
     - **Peptide**
     - **RNA**
     - **DNA**
     - **Small Molecule**
   * - :py:class:`seqme.models.Esm2`
     - |ok|
     - |ok|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.EsmFold`
     - |ok|
     - |no|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.ProstT5`
     - |ok|
     - |ok|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.RNA_FM`
     - |no|
     - |no|
     - |ok|
     - |no|
     - |no|
   * - :py:class:`seqme.models.GenaLM`
     - |no|
     - |no|
     - |no|
     - |ok|
     - |no|
   * - :py:class:`seqme.models.KmerFrequencyEmbedding`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |no|
   * - :py:class:`seqme.models.AliphaticIndex`
     - |ok|
     - |ok|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.Aromaticity`
     - |ok|
     - |ok|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.BomanIndex`
     - |ok|
     - |ok|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.Charge`
     - |ok|
     - |ok|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.Gravy`
     - |ok|
     - |ok|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.Hydrophobicity`
     - |ok|
     - |ok|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.HydrophobicMoment`
     - |ok|
     - |ok|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.InstabilityIndex`
     - |ok|
     - |ok|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.IsoelectricPoint`
     - |ok|
     - |ok|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.ProteinWeight`
     - |ok|
     - |ok|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.ThirdPartyModel`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.models.PCA`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
   * - :py:class:`seqme.models.Ensemble`
     - |ok|
     - |ok|
     - |ok|
     - |ok|
     - |ok|
