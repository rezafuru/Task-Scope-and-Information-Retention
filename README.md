# Task Scope and the Limits of Learned Compression

The paper studies how compression rates depend on the required predictions,
allowed errors, and information available to the encoder.

## Reproduce the figures

Requires Python 3.10 or newer. Run from the repository root.

```bash
pip install -e .
python scripts/make_figures.py
```

Uses the included results and images. Writes PDFs to `figures/`.

## Notebook

```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/quicklook.ipynb
```

## Experiments

Install model-fitting dependencies with `pip install -e '.[train]'`.
Use `--help` with the entry points below.

| Experiment | Entry points |
|---|---|
| Exact rates | [exact_rates.py](scripts/exact_rates.py) |
| Binary codes | [block_coding.py](scripts/block_coding.py) |
| Three-class codes | [rates](scripts/nonbinary_rates.py), [codebooks](scripts/nonbinary_codes.py), [learned selectors](scripts/nonbinary_learned.py) |
| ImageNet | [data](scripts/imagenet_data.py), [fitting](scripts/imagenet_fit.py), [assessment](scripts/imagenet_assess.py) |
| CIFAR-100 | [pipeline](scripts/cifar_pipeline.py), [analysis](scripts/cifar_analysis.py) |
| Taskonomy | [training](scripts/taskonomy_codec.py), [decoders](scripts/taskonomy_decoders.py), [paired analysis](scripts/taskonomy_paired.py), [analysis](scripts/taskonomy_analysis.py) |
| Species pair | [instructions](taskscope/pilot/README.md) |

Supply image datasets and trained study checkpoints separately. Recomputing
CIFAR and Taskonomy confidence intervals requires external per-image records.
Retained results and hashes are listed in [results/MANIFEST.json](results/MANIFEST.json).

## Citation

[[Preprint]](https://arxiv.org/abs/2609.37575v1)
```bibtex
@misc{furutanpey2026taskscopeinformationretention,
      title={On Task Scope and Information Retention in Source Coding}, 
      author={Alireza Furutanpey and Kerstin Bunte},
      year={2026},
      eprint={2609.37575},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/2609.37575}, 
}
```

MIT licence. See [LICENSE](LICENSE).
