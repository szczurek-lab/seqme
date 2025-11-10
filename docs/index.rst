seqme
=====
**seqme** is a modular and extendable python library containing model-agnostic metrics for evaluating biological sequence designs. It enables benchmarking and comparison of generative models for small molecules, DNA, RNA, peptides, and proteins.

**Key features**:

- **Metrics**: A collection of sequence-, embedding-, and property-based metrics for evaluating generative models designs.
- **Models**: Out-of-the-box, pre-trained property and embedding models for small molecules, DNA, RNA, peptides, and proteins.
- **Visualizations**: Functionality to display metric results from single-shot and iterative optimization methods as tables and plots.

*Is a metric or model missing?* seqme's modular metric and third-party model interfaces make adding your own easy.


Quick start
-----------

Install seqme and the protein language model, ESM-2.

.. code-block:: bash

    pip install "seqme[esm2]"


Run in a Jupyter notebook:

.. code-block:: python

    import seqme as sm

    sequences = {
        "Random": ["MKQW", "RKSPL"],
        "UniProt": ["KKWQ", "RKSPL", "RASD"],
        "HydrAMP": ["MMRK", "RKSPL", "RRLSK", "RRLSK"],
    }

    cache = sm.Cache(
        models={"esm2": sm.models.ESM2(
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
   contributing

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: About
   
   citing
   GitHub <https://github.com/szczurek-lab/seqme>



