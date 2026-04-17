"""
End-to-end regression test for the registration direction convention.

Loads the ground-truth transform JSON through the same code path used by the
UI callback (_apply_transform_json_str path: read JSON → AffineTransform →
apply_transform), then runs overlap computation and verifies the planted
statistical effects survive.

This test will catch any re-introduction of the direction-convention bug:
if the matrix is accidentally inverted (or stored as the forward ST→MEA
transform), the registered MEA locations will land in the wrong place and
the Pvalb correlation will collapse.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from skimage.transform import AffineTransform

SYNTH_DIR = Path(__file__).parent.parent / "synthetic_data"
SKIP_REASON = "Synthetic data not generated yet; run generate_synthetic_data.py first"


@pytest.mark.skipif(
    not (SYNTH_DIR / "synthetic_slide_tags.h5ad").exists(),
    reason=SKIP_REASON,
)
def test_gt_json_applies_directly_to_mea():
    """Loading the GT JSON and applying it directly must land MEA in ST space.

    This is the canonical smoke-test for the direction convention.  If the
    stored matrix is ST→MEA (the old bug) the registered locations will be
    far from the cells and the correlation will fail.
    """
    import scipy.sparse as sp

    from data.loaders import load_slide_tags, load_mea
    from analysis.registration import apply_transform
    from analysis.association import compute_overlap
    from analysis.stats import gene_firing_correlation

    st = load_slide_tags(
        str(SYNTH_DIR / "synthetic_slide_tags.h5ad"),
        spatial_units="μm",
        cell_type_col="cell_type",
    )
    mea = load_mea(str(SYNTH_DIR / "synthetic_mea"))

    # Load via the same code path as the callback: read JSON, build transform,
    # apply directly (no np.linalg.inv anywhere).
    gt_data = json.loads((SYNTH_DIR / "ground_truth_transform.json").read_text())
    transform = AffineTransform(matrix=np.array(gt_data["matrix"]))
    mea_locs_registered = apply_transform(mea.locations, transform)

    # Build expression for Pvalb
    idx = st.adata.var_names.get_loc("Pvalb")
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
        expression_matrix=col[:, None].astype(np.float32),
        gene_names=["Pvalb"],
        cell_labels=st.cell_types.tolist() if st.cell_types is not None else None,
    )

    pvalb_result = gene_firing_correlation(summaries, "Pvalb")

    assert pvalb_result.rho > 0.3, (
        f"Pvalb correlation too weak: ρ = {pvalb_result.rho:.3f}. "
        "Expected > 0.3. Check that the GT JSON stores the MEA→ST matrix."
    )
    assert pvalb_result.p_value < 0.01, (
        f"Pvalb correlation not significant: p = {pvalb_result.p_value:.4f}. "
        "Expected p < 0.01."
    )

    print(f"Pvalb rho = {pvalb_result.rho:.3f}, p = {pvalb_result.p_value:.4f}")


@pytest.mark.skipif(
    not (SYNTH_DIR / "synthetic_slide_tags.h5ad").exists(),
    reason=SKIP_REASON,
)
def test_gt_json_celltype_kruskal_wallis():
    """KW test must reject null (Pvalb vs Pyramidal firing rates differ) after
    registration via the direct-apply code path.
    """
    import scipy.sparse as sp

    from data.loaders import load_slide_tags, load_mea
    from analysis.registration import apply_transform
    from analysis.association import compute_overlap
    from analysis.stats import celltype_firing_comparison

    st = load_slide_tags(
        str(SYNTH_DIR / "synthetic_slide_tags.h5ad"),
        spatial_units="μm",
        cell_type_col="cell_type",
    )
    mea = load_mea(str(SYNTH_DIR / "synthetic_mea"))

    gt_data = json.loads((SYNTH_DIR / "ground_truth_transform.json").read_text())
    transform = AffineTransform(matrix=np.array(gt_data["matrix"]))
    mea_locs_registered = apply_transform(mea.locations, transform)

    summaries, _ = compute_overlap(
        cell_coords=st.coords,
        unit_coords=mea_locs_registered,
        unit_ids=mea.unit_ids,
        firing_rates=mea.firing_rates,
        radius=60.0,
        mode="hard",
        cell_labels=st.cell_types.tolist() if st.cell_types is not None else None,
    )

    result = celltype_firing_comparison(summaries, min_fraction=0.5)

    assert result.kw_p_value < 0.05, (
        f"KW test not significant: H = {result.kw_statistic:.3f}, "
        f"p = {result.kw_p_value:.4f}. Expected p < 0.05."
    )
    assert "Pvalb" in result.groups or "Pyramidal" in result.groups, (
        f"Expected Pvalb or Pyramidal group; got {list(result.groups.keys())}"
    )

    print(
        f"KW H = {result.kw_statistic:.3f}, p = {result.kw_p_value:.4f} | "
        f"groups: {list(result.groups.keys())}"
    )
