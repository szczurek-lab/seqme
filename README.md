<p align="left">
    <img src="docs/_static/logo_title.svg" alt="seqme logo" width="30%">
</p>

[![PyPI](https://img.shields.io/pypi/v/seqme.svg)](https://pypi.org/project/seqme/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/szczurek-lab/seqme)](https://opensource.org/license/bsd-3-clause)
[![Read the Docs](https://img.shields.io/readthedocs/seqme)](https://seqme.readthedocs.io/)


**seqme** is a modular and highly extendable python library containing model-agnostic metrics for evaluating biological sequences.

## Installation

You need to have Python 3.10 or newer installed on your system. To install the base package do:

```bash
$ pip install seqme
```

To also install domain-specific models, add extras specifiers.
Read the [docs](https://seqme.readthedocs.io/en/latest/api/models_index.html) for more information on the supported models.

## Quick start

Install seqme and the protein language model, ESM-2.

```bash
$ pip install seqme[esm2]
```

Run in a Jupyter notebook:

```python
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
sm.show_table(df) # Note: Will only display the table in a notebook.
```

Read the [docs](https://seqme.readthedocs.io/en/latest/tutorials/index.html) for more tutorials and examples.

## Citation

Preprint is coming soon.
