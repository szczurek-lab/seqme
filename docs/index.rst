pepme
=====
**pepme** is a modular open-source Python package containing model-agnostic metrics for evaluating peptides.

Quick start
-----------
.. code-block:: python

    from pepme import compute_metrics, show_table, FeatureCache
    from pepme.metrics import Uniqueness, Novelty, FID
    from pepme.models.embeddings import ESM2

    sequences = {
        "Random": ["MKQW", "RKSPL"],
        "UniProt": ["KKWQ", "RKSPL", "RASD"],
        "HydrAMP": ["MMRK", "RKSPL", "RRLSK", "RRLSK"],
    }

    cache = FeatureCache(
        models={"esm2": ESM2(model_name="esm2_t6_8M_UR50D", batch_size=256, device="cpu")}
    )

    metrics = [
        Uniqueness(),
        Novelty(reference=sequences["UniProt"], reference_name="UniProt"),
        FID(reference=sequences["Random"], embedder=cache.model("esm2")),
    ]

    df = compute_metrics(sequences, metrics)
    show_table(df)


.. toctree::
   :maxdepth: 3
   :hidden:
   :caption: General

   installation
   tutorials/index
   api
   contributing_guide

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: About

   GitHub <https://github.com/szczurek-lab/pepme>



