# Slide-tags × MEA Spatial Overlap Tool

A web tool for jointly analyzing Slide-tags spatial transcriptomics and MEA electrophysiology data from the same tissue section.

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)

## Setup

```bash
cd slidetags_mea_overlap
uv sync
```

## Run

```bash
uv run python app.py
```

Open http://127.0.0.1:8050 in your browser.

## Usage (v1)

1. **Load** — enter paths to your `.h5ad` file and SpikeInterface `SortingAnalyzer` folder, select spatial units, optionally name the cell-type column, then click **Load datasets**.
2. **Visualize** — explore side-by-side scatter plots; color Slide-tags by cell type or gene expression, MEA by firing rate.
3. **Register** — click ≥3 matched landmark pairs (alternating between plots), then click **Compute / update transform**.
4. **Overlap & Stats** — set footprint radius, select genes, compute overlap → view summary table, gene–firing correlations, cell-type comparisons.
5. **Export** — download summary CSV, transform JSON, or current plot.

## Run tests

```bash
uv run pytest
```
