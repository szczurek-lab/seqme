seqme
=====
**seqme** is a modular and highly extendable python library containing model-agnostic metrics for evaluating biological sequences.



Quick start
-----------

Let's compute a few metrics on example sequences.

.. code-block:: python

    import seqme as sm

    sequences = {
        "Random": ["MKQW", "RKSPL"],
        "UniProt": ["KKWQ", "RKSPL", "RASD"],
        "HydrAMP": ["MMRK", "RKSPL", "RRLSK", "RRLSK"],
    }

    cache = sm.ModelCache(
        models={"esm2": sm.models.Esm2(
            model_name="facebook/esm2_t6_8M_UR50D", batch_size=256, device="cpu")
        }
    )

    metrics = [
        sm.metrics.Uniqueness(),
        sm.metrics.Novelty(reference=sequences["UniProt"], reference_name="UniProt"),
        sm.metrics.FBD(reference=sequences["Random"], embedder=cache.model("esm2")),
    ]

    df = sm.compute_metrics(sequences, metrics)
    sm.show_table(df)


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



