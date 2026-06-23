# notebooks

Exploration / utility notebooks that sit on top of the `ppsync` package.

## Setup

The notebook deps live in the project's Poetry environment (repo
`pyproject.toml`), in the optional `notebooks` group — not a separate
requirements file:

```bash
# from the repo root
poetry install --with notebooks      # ppsync + jupyterlab + matplotlib
poetry run jupyter lab notebooks/
```

The notebooks add `<repo>/src` to `sys.path`, so they also import `ppsync`
whether or not the package is installed, as long as you launch from the repo
(or from `notebooks/`).

## Notebooks

| Notebook | What it does |
|---|---|
| `explore_mert_embeddings.ipynb` | Build the `.npz` prediction caches (single + batch — the same artifact `ppsync-preprocess` produces and `ppsync-align` / `tools/benchmark.py` consume) and explore the MERT embeddings inside one: window self-similarity (song structure), per-slide prototype similarity (repeat ambiguity), a NumPy-only 2-D PCA projection, and the before/after effect of contrastive normalization. |

## Tip

Notebooks accumulate cell outputs that are noisy in git diffs.  Clear outputs
before committing (`poetry run jupyter nbconvert --clear-output --inplace
<nb>.ipynb`, or Kernel → Restart & Clear Output).  `.ipynb_checkpoints/` is
gitignored.
