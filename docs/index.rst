pepme
=====
Metrics for evaluating generated peptides.

Quick start
-----------
.. code-block:: python

    from pepme import 
    from pepme.metrics import Uniqueness, Novelty, FID

    sequences = {
        "Random": ["MKQW", "RKSPL"],
        "UniProt": ["KKWQ", "RKSPL", "RASD"],
        "HydrAMP": ["MMRK", "RKSPL", "RRLSK", "RRLSK"],
    }
    
    metrics = [
        Uniqueness(),
        Novelty(reference=sequences["UniProt"], reference_name="UniProt"),
        FID(reference=sequences["Random"], embedder=my_embedder),
    ]

    df = compute_metrics(metrics, sequneces)
    show_table(df)


.. toctree::
    :maxdepth: 3
    :hidden:

    installation
    tutorials/index
    api
    contributing_guide


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
