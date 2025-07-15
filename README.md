# pepme

<p align="center">
  <img src="docs/_static/logo.svg" alt="pepme logo" width="15%">
</p>

**pepme** is a modular and highly extendable python library containing model-agnostic metrics for evaluating peptides.

## Installation

You need to have Python 3.10 or newer installed on your system.

```bash
$ pip install git+https://github.com/szczurek-lab/pepme.git
```

## Quick start

```python
from pepme import compute_metrics, show_table, FeatureCache
from pepme.metrics import Uniqueness, Novelty, FBD
from pepme.models.embeddings import Esm2

sequences = {
    "Random": ["MKQW", "RKSPL"],
    "UniProt": ["KKWQ", "RKSPL", "RASD"],
    "HydrAMP": ["MMRK", "RKSPL", "RRLSK", "RRLSK"],
}

cache = FeatureCache(
    models={"esm2": Esm2(model_name="facebook/esm2_t6_8M_UR50D", batch_size=256, device="cpu")}
)

metrics = [
    Uniqueness(),
    Novelty(reference=sequences["UniProt"], reference_name="UniProt"),
    FBD(reference=sequences["Random"], embedder=cache.model("esm2")),
]

df = compute_metrics(sequences, metrics)
show_table(df)
```
