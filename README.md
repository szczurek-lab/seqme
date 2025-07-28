<p align="left">
    <img src="docs/_static/logo_title.svg" alt="seqme logo" width="30%">
</p>

**seqme** is a modular and highly extendable python library containing model-agnostic metrics for biological sequences.

## Installation

You need to have Python 3.10 or newer installed on your system.

```bash
$ pip install git+https://github.com/szczurek-lab/seqme.git
```

## Quick start

```python
from seqme import compute_metrics, show_table, FeatureCache
from seqme.metrics import Uniqueness, Novelty, FBD
from seqme.models import Esm2

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
