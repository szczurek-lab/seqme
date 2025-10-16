seqme
=====
**seqme** is a modular and highly extendable python library containing model-agnostic metrics for evaluating biological sequences.


Quick start
-----------

Install seqme and the protein language model, ESM-2.

.. code-block:: bash

    pip install seqme[esm2]


Run in a Jupyter notebook:

.. code-block:: python

    import seqme as sm

    sequences = {
        "Random": ["MKQW", "RKSPL"],
        "UniProt": ["KKWQ", "RKSPL", "RASD"],
        "HydrAMP": ["MMRK", "RKSPL", "RRLSK", "RRLSK"],
    }

    cache = sm.Cache(
        models={"esm2": sm.models.Esm2(
            model_name="facebook/esm2_t6_8M_UR50D", batch_size=256, device="cpu")
        }
    )

    metrics = [
        sm.metrics.Uniqueness(),
        sm.metrics.Novelty(reference=sequences["UniProt"]),
        sm.metrics.FBD(reference=sequences["Random"], embedder=cache.model("esm2")),
    ]

    df = sm.evaluate(sequences, metrics)
    sm.show(df) # Note: Will only display the table in a notebook.


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
   
   citing
   GitHub <https://github.com/szczurek-lab/seqme>



