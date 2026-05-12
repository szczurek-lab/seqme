Models
======

Models can be viewed as sequence transforms or measures. They operate on a single entity—typically a biological sequence (or, in some cases, a 3D structure) - and map it to another representation or compute a value.

This distinguishes them from metrics: while a measure assigns a value to a single entity, a metric quantifies a relationship (e.g., distance or similarity) between two or more entities.

In practice, models transform biological sequences into feature representations or derive specific properties from them.

Overview
--------
The ``seqme`` library provides a suite of models that convert biological sequences into numerical representations. These models fall into two main categories:

* **Embedding models** — map sequences to fixed-length vector representations.
* **Property models** — compute scalar values or descriptors such as charge, weight, or hydrophobicity.

Additional third-party models are available at: https://github.com/szczurek-lab/seqme-thirdparty.

Models
------
.. autosummary::
    :toctree:
    :nosignatures:

    seqme.models.ESM2
    seqme.models.ESMFold
    seqme.models.RNAFM
    seqme.models.GENALM
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

    seqme.models.LogP
    seqme.models.QED
    seqme.models.SAScore
    

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
   * - :py:class:`seqme.models.ESM2`
     - |ok|
     - |ok|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.ESMFold`
     - |ok|
     - |no|
     - |no|
     - |no|
     - |no|
   * - :py:class:`seqme.models.RNAFM`
     - |no|
     - |no|
     - |ok|
     - |no|
     - |no|
   * - :py:class:`seqme.models.GENALM`
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
   * - :py:class:`seqme.models.LogP`
     - |no|
     - |no|
     - |no|
     - |no|
     - |ok|
   * - :py:class:`seqme.models.QED`
     - |no|
     - |no|
     - |no|
     - |no|
     - |ok|
   * - :py:class:`seqme.models.SAScore`
     - |no|
     - |no|
     - |no|
     - |no|
     - |ok|
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
