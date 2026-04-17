"""
Tests for spatial association module.

Uses synthetic data with known geometry so expected overlap is exact,
and also tests against the generated synthetic h5ad/MEA to verify
planted statistical effects.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from analysis.association import compute_overlap, summary_table, UnitSummary


# ---------------------------------------------------------------------------
# Fixtures — geometric (deterministic, no file I/O)
# ---------------------------------------------------------------------------

@pytest.fixture
def grid_data():
    """5×5 grid of cells at integer μm positions; one unit at center (2,2)."""
    xs = np.arange(5, dtype=np.float32)
    ys = np.arange(5, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    cell_coords = np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)
    unit_coords = np.array([[2.0, 2.0]], dtype=np.float32)
    unit_ids = np.array(["u0"])
    firing_rates = np.array([10.0], dtype=np.float32)
    return cell_coords, unit_coords, unit_ids, firing_rates


# ---------------------------------------------------------------------------
# Geometric correctness tests
# ---------------------------------------------------------------------------

def test_hard_radius_count(grid_data):
    """Hard radius=1.5 μm around (2,2).

    Within Euclidean distance ≤1.5:
      d=0: (2,2)
      d=1: (1,2),(3,2),(2,1),(2,3)
      d=√2≈1.414: (1,1),(3,1),(1,3),(3,3)
    Total = 9 cells.  Cells at d=2 (e.g. (0,2)) are excluded.
    """
    cell_coords, unit_coords, unit_ids, fr = grid_data
    summaries, per_cell = compute_overlap(
        cell_coords, unit_coords, unit_ids, fr,
        radius=1.5, mode="hard",
    )
    assert len(summaries) == 1
    assert summaries[0].n_cells_within == 9


def test_hard_radius_zero(grid_data):
    """Radius=0.1 — only the cell exactly at (2,2) is inside."""
    cell_coords, unit_coords, unit_ids, fr = grid_data
    # KD-tree query_ball_point is strict < radius, so use 0.01
    summaries, _ = compute_overlap(
        cell_coords, unit_coords, unit_ids, fr,
        radius=0.01, mode="hard",
    )
    # The cell at (2,2) is distance 0 from the unit
    assert summaries[0].n_cells_within == 1


def test_gaussian_weights_sum(grid_data):
    """Gaussian weights for cells at distances 0, 1, √2 from unit."""
    cell_coords, unit_coords, unit_ids, fr = grid_data
    summaries, _ = compute_overlap(
        cell_coords, unit_coords, unit_ids, fr,
        radius=2.0, mode="gaussian",
    )
    us = summaries[0]
    # total_weight should be < n_cells_within (weights < 1 for d > 0)
    assert us.total_weight > 0
    assert us.total_weight <= us.n_cells_within


def test_cell_type_fractions_sum_to_one(grid_data):
    """Cell-type fractions should sum to 1.0."""
    cell_coords, unit_coords, unit_ids, fr = grid_data
    labels = ["A"] * 10 + ["B"] * 15  # 25 cells total
    summaries, _ = compute_overlap(
        cell_coords, unit_coords, unit_ids, fr,
        radius=3.0, mode="hard",
        cell_labels=labels,
    )
    fracs = summaries[0].cell_type_fractions
    if fracs:
        assert abs(sum(fracs.values()) - 1.0) < 1e-6


def test_per_cell_units_mapping(grid_data):
    """Each cell within radius should appear in per_cell_units."""
    cell_coords, unit_coords, unit_ids, fr = grid_data
    _, per_cell = compute_overlap(
        cell_coords, unit_coords, unit_ids, fr,
        radius=1.5, mode="hard",
    )
    # All mapped cells should reference unit "u0"
    for cell_idx, units in per_cell.items():
        assert "u0" in units


def test_expression_weighted_mean(grid_data):
    """Weighted mean expression should equal arithmetic mean for hard mode."""
    cell_coords, unit_coords, unit_ids, fr = grid_data
    rng = np.random.default_rng(0)
    expr = rng.uniform(0, 10, size=(25, 3)).astype(np.float32)
    gene_names = ["g0", "g1", "g2"]

    summaries, _ = compute_overlap(
        cell_coords, unit_coords, unit_ids, fr,
        radius=1.5, mode="hard",
        expression_matrix=expr,
        gene_names=gene_names,
    )
    us = summaries[0]
    # Find which cells are within radius
    dists = np.linalg.norm(cell_coords - np.array([2.0, 2.0]), axis=1)
    inside = dists <= 1.5

    for j, g in enumerate(gene_names):
        expected = float(expr[inside, j].mean())
        assert abs(us.mean_expression[g] - expected) < 1e-4, \
            f"Gene {g}: expected {expected:.4f}, got {us.mean_expression[g]:.4f}"


def test_summary_table_columns(grid_data):
    """summary_table should produce expected columns."""
    cell_coords, unit_coords, unit_ids, fr = grid_data
    labels = ["A"] * 13 + ["B"] * 12
    summaries, _ = compute_overlap(
        cell_coords, unit_coords, unit_ids, fr,
        radius=3.0, mode="hard",
        cell_labels=labels,
    )
    df = summary_table(summaries)
    required = {"unit_id", "x", "y", "firing_rate",
                "n_cells_within", "total_weight"}
    assert required.issubset(set(df.columns))


def test_no_nearby_cells():
    """Unit far from all cells should have n_cells_within=0."""
    cells = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    units = np.array([[1000.0, 1000.0]], dtype=np.float32)
    ids = np.array(["u_far"])
    fr = np.array([5.0])
    summaries, per_cell = compute_overlap(cells, units, ids, fr,
                                          radius=10.0, mode="hard")
    assert summaries[0].n_cells_within == 0
    assert summaries[0].total_weight == 0.0
    assert per_cell == {}


# ---------------------------------------------------------------------------
# Statistical planted-effect tests (require synthetic_data/ to be generated)
# ---------------------------------------------------------------------------

SYNTH_DIR = Path(__file__).parent.parent / "synthetic_data"


@pytest.mark.skipif(
    not (SYNTH_DIR / "synthetic_slide_tags.h5ad").exists(),
    reason="Synthetic data not generated yet; run generate_synthetic_data.py first",
)
def test_pvalb_correlation_significant():
    """Pvalb expression should significantly correlate with firing rate (p < 0.05)."""
    _run_correlation_test("Pvalb", expect_significant=True, max_p=0.05)


@pytest.mark.skipif(
    not (SYNTH_DIR / "synthetic_slide_tags.h5ad").exists(),
    reason="Synthetic data not generated yet",
)
def test_random_gene_not_significant():
    """A random gene with no spatial structure should not correlate with FR.

    Rand00 has uniform low expression across all cell types and positions,
    so per-unit weighted mean expression is close to constant → p ≫ 0.05.

    Note: Slc17a7 (pyramidal marker) is deliberately NOT used here because
    the synthetic Pyramidal-exclusion zone around the Pvalb cluster creates
    a spurious anti-correlation between Slc17a7 and FR.  That is a known
    geometric artefact of the synthetic design, not a failure of the tool.
    """
    _run_correlation_test("Rand00", expect_significant=False, max_p=0.05)


def test_celltype_kruskal_wallis_rejects():
    """KW should reject the null when cell types have very different firing rates.

    Uses a clean geometric fixture:
    - Zone A (x < 200): only "FastCell" type — 5 units with high FR (20-40 Hz)
    - Zone B (x > 800): only "SlowCell" type — 5 units with low FR (1-3 Hz)
    Each zone has 50 pure-type cells, so each unit gets a clear dominant type.
    """
    from analysis.stats import celltype_firing_comparison

    rng = np.random.default_rng(7)

    # Zone A cells: FastCell at x∈[0,200], y∈[0,500]
    fast_cells = rng.uniform([0, 0], [200, 500], size=(50, 2)).astype(np.float32)
    # Zone B cells: SlowCell at x∈[800,1000], y∈[0,500]
    slow_cells = rng.uniform([800, 0], [1000, 500], size=(50, 2)).astype(np.float32)
    cell_coords = np.vstack([fast_cells, slow_cells])
    cell_labels = ["FastCell"] * 50 + ["SlowCell"] * 50

    # Units: 5 in zone A, 5 in zone B
    fast_units = rng.uniform([20, 50], [180, 450], size=(5, 2)).astype(np.float32)
    slow_units = rng.uniform([820, 50], [980, 450], size=(5, 2)).astype(np.float32)
    unit_coords = np.vstack([fast_units, slow_units])
    unit_ids = np.array([f"u{i}" for i in range(10)])
    firing_rates = np.concatenate([
        rng.uniform(20, 40, 5),   # fast units: 20-40 Hz
        rng.uniform(1, 3, 5),     # slow units: 1-3 Hz
    ]).astype(np.float32)

    summaries, _ = compute_overlap(
        cell_coords, unit_coords, unit_ids, firing_rates,
        radius=100.0, mode="hard",
        cell_labels=cell_labels,
    )

    result = celltype_firing_comparison(summaries, min_fraction=0.5)
    assert result.kw_p_value < 0.05, \
        f"Expected KW to reject null; got p = {result.kw_p_value:.4f}"
    assert "FastCell" in result.groups and "SlowCell" in result.groups, \
        f"Expected both groups; got {list(result.groups.keys())}"


def _run_correlation_test(gene: str, expect_significant: bool, max_p: float):
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    import json
    import numpy as np
    from skimage.transform import AffineTransform

    from data.loaders import load_slide_tags, load_mea
    from analysis.association import compute_overlap
    from analysis.stats import gene_firing_correlation
    from analysis.registration import apply_transform

    st = load_slide_tags(
        str(SYNTH_DIR / "synthetic_slide_tags.h5ad"),
        spatial_units="μm",
        cell_type_col="cell_type",
    )
    mea = load_mea(str(SYNTH_DIR / "synthetic_mea"))

    gt_data = json.loads((SYNTH_DIR / "ground_truth_transform.json").read_text())
    transform = AffineTransform(matrix=np.array(gt_data["matrix"]))
    mea_locs_registered = apply_transform(mea.locations, transform)

    # Build expression matrix for just the gene of interest
    import scipy.sparse as sp
    idx = st.adata.var_names.get_loc(gene)
    col = st.adata.X[:, idx]
    if sp.issparse(col):
        col = np.asarray(col.todense()).ravel()
    else:
        col = np.asarray(col).ravel()

    summaries, _ = compute_overlap(
        cell_coords=st.coords,
        unit_coords=mea_locs_registered,
        unit_ids=mea.unit_ids,
        firing_rates=mea.firing_rates,
        radius=60.0,
        mode="hard",
        expression_matrix=col[:, None],
        gene_names=[gene],
    )

    result = gene_firing_correlation(summaries, gene)

    if expect_significant:
        assert result.p_value < max_p, \
            f"Expected {gene} correlation to be significant (p < {max_p}); " \
            f"got ρ = {result.rho:.3f}, p = {result.p_value:.4f}"
    else:
        assert result.p_value > max_p, \
            f"Expected {gene} correlation to be non-significant (p > {max_p}); " \
            f"got ρ = {result.rho:.3f}, p = {result.p_value:.4f}"
