# CLAUDE.md

Context for Claude Code sessions on this project. Read this before editing.

## What this is

A Dash web app that spatially overlaps two neuroscience datasets from the same
tissue section:
- **Slide-tags** — single-cell spatial transcriptomics (.h5ad, cells × genes +
  2D coords in μm + cell-type labels)
- **MEA** — multi-electrode array electrophysiology (SpikeInterface
  SortingAnalyzer, units with 2D locations + firing rates)

The tool registers the two coordinate systems from manual landmarks (affine),
computes which cells fall inside each unit's spatial footprint, and runs two
statistics: Spearman gene–firing correlation and Kruskal-Wallis cell-type
firing comparison. Four Dash tabs: Visualize, Register, Overlap & Stats,
Export.

The user is not a Python programmer. They have strong biology knowledge but
not PhD-level stats. Explain statistical choices when they come up and flag
when results need expert interpretation.

## Critical invariant — transform direction

**Every affine matrix stored on disk or in a Dash store maps MEA (moving) →
Slide-tags (fixed).** Apply it directly to MEA unit coordinates to place them
in Slide-tags space. No inversion anywhere in the production code path.

This convention was broken once (the ground-truth JSON originally stored
ST→MEA and the callback silently applied it forward to MEA coords, producing
~100 μm errors and killing the planted Pvalb correlation). The fix unified
everything on MEA→ST. The regression tests in `tests/test_registration_end_to_end.py`
exist specifically to catch any re-introduction of the bug. Do not "simplify"
them away.

If you find yourself wanting to call `np.linalg.inv(...)` on a transform
matrix anywhere outside `generate_synthetic_data.py`, stop. That's almost
certainly the bug coming back.

## Repo layout

- `app.py` — Dash entry point
- `analysis/`
  - `registration.py` — affine fit from landmarks; `save_transform` and
    `_make_transform_dict` define the canonical JSON schema
  - `association.py` — `compute_overlap` (cKDTree, hard + gaussian modes)
  - `stats.py` — Spearman, Kruskal-Wallis, BH correction
- `data/`
  - `loaders.py` — reads .h5ad and SpikeInterface SortingAnalyzer
  - `schema.py` — `SlideTagsData`, `MEAData`, `RegisteredPair` dataclasses
- `ui/`
  - `layout.py` — Dash layout
  - `callbacks.py` — all callbacks; `_apply_transform_json_str` is the
    single load path for transform JSONs
- `generate_synthetic_data.py` — generates the full synthetic dataset plus
  `ground_truth_transform.json`. Contains the only legitimate use of
  `np.linalg.inv` in the codebase, for writing the MEA→ST JSON.
- `synthetic_data/` — generated artefacts; regenerate with
  `uv run python generate_synthetic_data.py`
- `tests/` — pytest suite; 17 tests currently pass

## Commands

```bash
# Install / sync
uv sync

# Run the app
uv run python app.py     # → http://127.0.0.1:8050

# Regenerate synthetic data (includes a roundtrip assertion on the JSON)
uv run python generate_synthetic_data.py

# Run tests
uv run pytest
uv run pytest -v         # verbose
uv run pytest tests/test_registration_end_to_end.py  # regression tests only
```

## Tech stack

Python 3.13, Dash/Plotly, anndata, scanpy, spikeinterface 0.104.1,
scikit-image 0.26, scipy, uv + pyproject.toml.

`cKDTree` is used in `association.py`; SciPy is deprecating it in favor of
`KDTree` but they're interchangeable for our calls. Don't migrate unless
asked.

## What has been validated

- Geometric correctness of `compute_overlap` (grid tests)
- Registration math recovers known transforms to 1e-4
- On synthetic data with planted effects: Pvalb Spearman ρ ≈ 0.46, p < 0.01
  at r=50μm; KW separates Pyramidal vs Pvalb groups at p < 0.05
- End-to-end regression test loads the GT JSON through the real callback
  path and verifies the above

## What has NOT been validated

- Any real paired Slide-tags + MEA data (none publicly exists as of 2026)
- Non-affine tissue deformation (real tissue shrinks/tears; may need
  thin-plate spline)
- Large datasets (>5k cells, >500 units) — performance untested
- Robustness to realistic registration noise (planned sensitivity analysis)
- Multiple-testing correction across genes in gene-FR correlation
- Spatial autocorrelation adjustment (nearby units see overlapping cells,
  violating independence; Spearman p-values are optimistic)

Be honest about these limits if the user asks about scientific validity.

## Conventions to follow

- Don't add dependencies without asking.
- Don't rename public APIs, Dash component IDs, or store keys.
- Changes that touch the transform convention need an updated regression test.
- When modifying `compute_overlap` or `_apply_transform_json_str`, run the
  full test suite and report the Pvalb ρ / p values from the end-to-end test.
- If a change would require `np.linalg.inv` anywhere outside
  `generate_synthetic_data.py`, stop and explain why before proceeding.
- Shortcut: after any non-trivial change, run `uv run pytest` and report
  results inline.

## Known future work (user's priorities)

1. Validate on real data — likely Emery et al. 2025 Zenodo datasets
   (Visium + LFP, structurally different from Slide-tags + spike-sorted; any
   run on this data is a "demonstration", not true validation).
2. Add sparsity-constrained NMF for joint gene–activity spatial patterns
   (inspired by MEA-seqX; adapt rather than copy).
3. UI polish — better error messages, loading spinners, help text.

Do not work on these unless the user explicitly asks. The main bug is fixed;
further changes should be user-directed.
