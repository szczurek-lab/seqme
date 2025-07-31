seqme
=====
**seqme** is a modular and highly extendable python library containing model-agnostic metrics for evaluating biological sequences.



Quick start
-----------

Let's compute a few metrics on example sequences.

.. code-block:: python

    from seqme import compute_metrics, show_table, ModelCache
    from seqme.metrics import Uniqueness, Novelty, FBD
    from seqme.models import Esm2

    sequences = {
        "Random": ["MKQW", "RKSPL"],
        "UniProt": ["KKWQ", "RKSPL", "RASD"],
        "HydrAMP": ["MMRK", "RKSPL", "RRLSK", "RRLSK"],
    }

    cache = ModelCache(
        models={"esm2": Esm2(model_name="facebook/esm2_t6_8M_UR50D", batch_size=256, device="cpu")}
    )

    metrics = [
        Uniqueness(),
        Novelty(reference=sequences["UniProt"], reference_name="UniProt"),
        FBD(reference=sequences["Random"], embedder=cache.model("esm2")),
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

   GitHub <https://github.com/szczurek-lab/seqme>



