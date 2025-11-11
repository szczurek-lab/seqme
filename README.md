<p align="left">
    <img src="https://raw.githubusercontent.com/szczurek-lab/seqme/main/docs/_static/logo_title.svg" alt="seqme logo" width="30%">
</p>

[![PyPI](https://img.shields.io/pypi/v/seqme.svg)](https://pypi.org/project/seqme/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/github/license/szczurek-lab/seqme)](https://opensource.org/license/bsd-3-clause)
[![Read the Docs](https://img.shields.io/readthedocs/seqme)](https://seqme.readthedocs.io/)

**seqme** is a modular and extendable python library containing model-agnostic metrics for evaluating biological sequence designs. It enables benchmarking and comparison of generative models for small molecules, DNA, RNA, peptides, and proteins.

**Key features**:

- **[Metrics](https://seqme.readthedocs.io/en/stable/api/metrics_index.html)**: A collection of sequence-, embedding-, and property-based metrics for evaluating generative models designs.
- **[Models](https://seqme.readthedocs.io/en/stable/api/models_index.html)**: Out-of-the-box, pre-trained property and embedding models for small molecules, DNA, RNA, peptides, and proteins.
- **[Visualizations](https://seqme.readthedocs.io/en/stable/api/core_index.html#visualization)**: Functionality to display metric results from single-shot and iterative optimization methods as tables and plots.

*Is a metric or model missing?* seqme's modular metric and third-party model interfaces make adding your own easy.

## Installation

You need to have Python 3.10 or newer installed on your system. To install the base package do:

```bash
$ pip install seqme
```

To install sequence-specific models as well, include the appropriate extras specifiers. 
Check the individual model [docs](https://seqme.readthedocs.io/en/stable/api/models_index.html) for installation instructions.

## Quick start

Install seqme and the protein language model, ESM-2.

```bash
$ pip install "seqme[esm2]"
```

Run in a Jupyter notebook:

```python
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
```

Check out the [docs](https://seqme.readthedocs.io/en/stable/tutorials/index.html) for in-depth tutorials and examples.

## Citation

If you use **seqme** in your research, consider citing our [publication](https://arxiv.org/abs/2511.04239):

```bibtex
@article{mollerlarsen2025seqme,
      title={seqme: a Python library for evaluating biological sequence design}, 
      author={Rasmus Møller-Larsen and Adam Izdebski and Jan Olszewski and Pankhil Gawade and Michal Kmicikiewicz and Wojciech Zarzecki and Ewa Szczurek},
      year={2025},
      eprint={2511.04239},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.04239}, 
}
```
