"""
Generate synthetic test data for the Slide-tags × MEA Spatial Overlap Tool.

Outputs (written to ./synthetic_data/):
  synthetic_slide_tags.h5ad       - 1000-cell Slide-tags dataset
  synthetic_mea/                  - SpikeInterface SortingAnalyzer folder
  ground_truth_transform.json     - known affine transform (rotation + translation)

Planted effects (verified by tests):
  - Pvalb expression in nearby cells correlates with firing rate (ρ ≈ 0.5–0.7)
  - Slc17a7 (pyramidal marker) shows weak / no correlation
  - Pvalb+ dominated units fire faster than pyramidal-dominated (effect size d > 1)

Run:
    uv run python generate_synthetic_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scipy.sparse as sp
import spikeinterface.core as sic
from probeinterface import Probe
from skimage.transform import AffineTransform
from spikeinterface.postprocessing.unit_locations import ComputeUnitLocations

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SEED = 42
RNG = np.random.default_rng(SEED)

OUT_DIR = Path(__file__).parent / "synthetic_data"
OUT_DIR.mkdir(exist_ok=True)

FIELD_W, FIELD_H = 1000.0, 500.0   # μm
N_CELLS = 1000
N_UNITS = 50
DURATION_S = 300.0                  # seconds of recording
FS = 30000.0                        # Hz sampling rate
N_MEA_CHANNELS = 32

# Ground-truth transform applied to MEA coords (MEA is "moving" frame)
GT_ROTATION_DEG = 8.0
GT_TRANSLATION = np.array([30.0, -20.0])   # μm

# ---------------------------------------------------------------------------
# Cell-type proportions and spatial layout
# ---------------------------------------------------------------------------

CELL_TYPES = {
    # Pyramidal cells: excluded from the Pvalb zone (y∈[355,425]) so that
    # Pvalb-dominated units actually exist.  y_sigma=0 → uniform outside zone.
    "Pyramidal":        {"frac": 0.65, "y_center": 250.0, "y_sigma": 0.0,   "x_uniform": True},
    # Pvalb cells: tight cluster at y≈390.
    # Pvalb density drives firing rate → clear Pvalb×FR correlation.
    "Pvalb":            {"frac": 0.05, "y_center": 390.0, "y_sigma": 18.0,  "x_uniform": True},
    "Sst":              {"frac": 0.03, "y_center": 110.0, "y_sigma": 20.0,  "x_uniform": True},
    "Astrocyte":        {"frac": 0.15, "y_center": 250.0, "y_sigma": 200.0, "x_uniform": True},
    "Oligodendrocyte":  {"frac": 0.12, "y_center": 250.0, "y_sigma": 200.0, "x_uniform": True},
}

# Marker genes: (mean_expression per cell type); unlisted types get baseline 0.2
GENE_MEANS: dict[str, dict[str, float]] = {
    "Slc17a7":  {"Pyramidal": 8.0,  "Pvalb": 0.2, "Sst": 0.2, "Astrocyte": 0.2, "Oligodendrocyte": 0.2},
    "Gad1":     {"Pyramidal": 0.2,  "Pvalb": 6.0, "Sst": 6.0, "Astrocyte": 0.2, "Oligodendrocyte": 0.2},
    "Gad2":     {"Pyramidal": 0.2,  "Pvalb": 5.0, "Sst": 5.0, "Astrocyte": 0.2, "Oligodendrocyte": 0.2},
    "Pvalb":    {"Pyramidal": 0.1,  "Pvalb": 9.0, "Sst": 0.3, "Astrocyte": 0.1, "Oligodendrocyte": 0.1},
    "Sst":      {"Pyramidal": 0.1,  "Pvalb": 0.3, "Sst": 8.0, "Astrocyte": 0.1, "Oligodendrocyte": 0.1},
    "Vip":      {"Pyramidal": 0.1,  "Pvalb": 0.1, "Sst": 0.1, "Astrocyte": 0.1, "Oligodendrocyte": 0.1},
    "Gfap":     {"Pyramidal": 0.2,  "Pvalb": 0.2, "Sst": 0.2, "Astrocyte": 7.0, "Oligodendrocyte": 0.5},
    "Mbp":      {"Pyramidal": 0.1,  "Pvalb": 0.1, "Sst": 0.1, "Astrocyte": 0.3, "Oligodendrocyte": 8.0},
    "Bdnf":     {"Pyramidal": 2.0,  "Pvalb": 1.5, "Sst": 1.5, "Astrocyte": 0.5, "Oligodendrocyte": 0.5},
    "Egr1":     {"Pyramidal": 3.0,  "Pvalb": 2.0, "Sst": 1.5, "Astrocyte": 0.5, "Oligodendrocyte": 0.3},
    "Fos":      {"Pyramidal": 2.5,  "Pvalb": 2.0, "Sst": 1.5, "Astrocyte": 0.3, "Oligodendrocyte": 0.2},
    "Arc":      {"Pyramidal": 2.0,  "Pvalb": 1.0, "Sst": 0.8, "Astrocyte": 0.2, "Oligodendrocyte": 0.1},
    "Homer1":   {"Pyramidal": 2.5,  "Pvalb": 1.0, "Sst": 0.8, "Astrocyte": 0.2, "Oligodendrocyte": 0.1},
    "Camk2a":   {"Pyramidal": 4.0,  "Pvalb": 0.5, "Sst": 0.5, "Astrocyte": 0.3, "Oligodendrocyte": 0.2},
    "Gria2":    {"Pyramidal": 3.5,  "Pvalb": 1.0, "Sst": 0.8, "Astrocyte": 0.3, "Oligodendrocyte": 0.2},
    "Npas4":    {"Pyramidal": 2.5,  "Pvalb": 1.5, "Sst": 1.0, "Astrocyte": 0.2, "Oligodendrocyte": 0.1},
}

# 14 random genes (uniform low expression)
RANDOM_GENES = [f"Rand{i:02d}" for i in range(14)]
ALL_GENES = list(GENE_MEANS.keys()) + RANDOM_GENES

# Unit type distribution (drives PLACEMENT; FR is driven by Pvalb density, see below)
UNIT_TYPES = {
    "Pyramidal_dominated": {"n": 30},
    "Pvalb_dominated":     {"n":  8},
    "Sst_dominated":       {"n":  5},
    "Mixed":               {"n":  7},
}

FOOTPRINT_RADIUS = 60.0  # μm — used for planting Pvalb ↔ FR correlation

# Firing-rate model:
#   FR_i = baseline_noise + pvalb_gain × n_pvalb_within_radius
# This makes Pvalb density the PRIMARY driver so the correlation is clear.
FR_BASELINE_LO = 1.0   # Hz (minimum noise floor)
FR_BASELINE_HI = 4.0   # Hz (maximum baseline — small to not dominate)
FR_PVALB_GAIN  = 2.5   # Hz added per Pvalb cell within footprint


# ---------------------------------------------------------------------------
# Step 1: Slide-tags
# ---------------------------------------------------------------------------

def make_cell_coords() -> tuple[np.ndarray, list[str]]:
    """Return (N, 2) coords in μm and parallel list of cell-type labels."""
    coords = []
    labels = []
    n_assigned = 0
    type_list = list(CELL_TYPES.keys())

    for i, (ct, cfg) in enumerate(CELL_TYPES.items()):
        n = N_CELLS if i == len(type_list) - 1 else int(round(cfg["frac"] * N_CELLS))
        # Prevent cumulative rounding error on last type
        if i == len(type_list) - 1:
            n = N_CELLS - n_assigned

        x = RNG.uniform(0, FIELD_W, size=n)
        if cfg["y_sigma"] == 0.0 and ct == "Pyramidal":
            # Uniform y EXCLUDING the Pvalb zone [355, 425].
            # This ensures Pvalb-dominated units exist near y≈390.
            PVALB_ZONE_LO, PVALB_ZONE_HI = 355.0, 425.0
            zone_width = PVALB_ZONE_HI - PVALB_ZONE_LO
            valid_height = FIELD_H - zone_width   # 430 μm
            raw_y = RNG.uniform(0, valid_height, size=n)
            # Map values above PVALB_ZONE_LO upward past the exclusion zone
            y = np.where(raw_y < PVALB_ZONE_LO, raw_y,
                         raw_y + zone_width)
        elif cfg["y_sigma"] == 0.0:
            y = RNG.uniform(0, FIELD_H, size=n)
        else:
            y = RNG.normal(cfg["y_center"], cfg["y_sigma"], size=n)
        y = np.clip(y, 0, FIELD_H)

        coords.append(np.column_stack([x, y]))
        labels.extend([ct] * n)
        n_assigned += n

    return np.vstack(coords).astype(np.float32), labels


def make_expression(labels: list[str]) -> np.ndarray:
    """Return (n_cells, n_genes) count matrix (negative binomial)."""
    n = len(labels)
    n_genes = len(ALL_GENES)
    X = np.zeros((n, n_genes), dtype=np.float32)

    for j, gene in enumerate(ALL_GENES):
        if gene in GENE_MEANS:
            means_by_type = GENE_MEANS[gene]
        else:
            # random gene: low uniform expression
            means_by_type = {ct: RNG.uniform(0.1, 0.5) for ct in CELL_TYPES}

        for i, ct in enumerate(labels):
            mu = means_by_type.get(ct, 0.2)
            # Negative binomial: mean=mu, dispersion r=2
            r = 2.0
            p = r / (r + mu)
            X[i, j] = RNG.negative_binomial(r, p)

    return X


def make_slide_tags() -> anndata.AnnData:
    print("Generating Slide-tags cells…")
    coords, labels = make_cell_coords()
    X = make_expression(labels)

    obs = pd.DataFrame({"cell_type": labels}, index=[f"cell_{i}" for i in range(N_CELLS)])
    var = pd.DataFrame(index=ALL_GENES)

    adata = anndata.AnnData(
        X=sp.csr_matrix(X),
        obs=obs,
        var=var,
    )
    adata.obsm["spatial"] = coords  # (N, 2) in μm
    return adata


# ---------------------------------------------------------------------------
# Step 2: MEA / SortingAnalyzer
# ---------------------------------------------------------------------------

def _gt_transform() -> AffineTransform:
    """Return the ground-truth affine transform (rotation + translation)."""
    angle = np.deg2rad(GT_ROTATION_DEG)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    matrix = np.array([
        [cos_a, -sin_a, GT_TRANSLATION[0]],
        [sin_a,  cos_a, GT_TRANSLATION[1]],
        [0.0,    0.0,   1.0],
    ])
    return AffineTransform(matrix=matrix)


def make_unit_layout() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Return:
      slide_tags_locs  - (n_units, 2) unit positions in Slide-tags space (μm)
      mea_locs         - (n_units, 2) unit positions in MEA space (after GT transform)
      unit_type_labels - parallel list of unit-type strings
    """
    locs_st = []
    unit_types = []

    # Place units to match cell-type spatial layout
    for utype, cfg in UNIT_TYPES.items():
        n = cfg["n"]
        if "Pyramidal" in utype:
            # Placed in pyramidal band (away from Pvalb cluster)
            x = RNG.uniform(50, FIELD_W - 50, n)
            y = RNG.normal(250, 25, n)
        elif "Pvalb" in utype:
            # Placed near Pvalb cell cluster at y≈390
            x = RNG.uniform(50, FIELD_W - 50, n)
            y = RNG.normal(390, 15, n)
        elif "Sst" in utype:
            x = RNG.uniform(50, FIELD_W - 50, n)
            y = RNG.normal(110, 20, n)
        else:  # Mixed
            x = RNG.uniform(50, FIELD_W - 50, n)
            y = RNG.uniform(50, FIELD_H - 50, n)

        y = np.clip(y, 10, FIELD_H - 10)
        locs_st.append(np.column_stack([x, y]))
        unit_types.extend([utype] * n)

    locs_st = np.vstack(locs_st).astype(np.float32)

    # Apply ground-truth transform to get MEA coordinates
    gt = _gt_transform()
    locs_mea = gt(locs_st.astype(np.float64)).astype(np.float32)

    return locs_st, locs_mea, unit_types


def make_firing_rates(unit_types: list[str], cell_coords: np.ndarray,
                      cell_labels: list[str], unit_locs_st: np.ndarray) -> np.ndarray:
    """
    Generate firing rates where Pvalb cell density is the PRIMARY driver.

    Model:
        FR_i = Uniform(FR_BASELINE_LO, FR_BASELINE_HI)
               + FR_PVALB_GAIN × n_pvalb_within_FOOTPRINT_RADIUS
               + small noise

    This ensures Spearman ρ(Pvalb expression, FR) ≈ 0.5–0.7 while
    Slc17a7 (expressed in ubiquitous Pyramidal cells) shows no correlation.
    """
    from scipy.spatial import cKDTree

    pvalb_mask = np.array([l == "Pvalb" for l in cell_labels])
    pvalb_coords = cell_coords[pvalb_mask]

    fr = np.zeros(N_UNITS, dtype=np.float32)
    tree = cKDTree(pvalb_coords) if len(pvalb_coords) > 0 else None

    for i, uloc in enumerate(unit_locs_st):
        baseline = float(RNG.uniform(FR_BASELINE_LO, FR_BASELINE_HI))

        n_pvalb = 0
        if tree is not None:
            nearby = tree.query_ball_point(uloc, r=FOOTPRINT_RADIUS)
            n_pvalb = len(nearby)

        noise = float(RNG.uniform(-0.5, 0.5))
        fr[i] = max(0.1, baseline + FR_PVALB_GAIN * n_pvalb + noise)

    return fr


def make_mea_sorting_analyzer(unit_locs_mea: np.ndarray,
                               firing_rates: np.ndarray) -> sic.SortingAnalyzer:
    """
    Build a SpikeInterface SortingAnalyzer with:
    - Synthetic recording (noise) with probe covering the MEA field
    - NumpySorting with spike trains matching requested firing rates
    - unit_locations extension pre-computed and overridden with known locations
    """
    # Probe: 32-channel grid covering the MEA field
    n_ch = N_MEA_CHANNELS
    ch_x = np.tile(np.linspace(0, FIELD_W, n_ch // 4), 4)
    ch_y = np.repeat(np.linspace(0, FIELD_H, 4), n_ch // 4)
    ch_locs = np.column_stack([ch_x, ch_y]).astype(float)

    probe = Probe(ndim=2, si_units="um")
    probe.set_contacts(positions=ch_locs, shapes="circle",
                       shape_params={"radius": 7})
    probe.set_device_channel_indices(np.arange(n_ch))

    # Recording
    rec = sic.generate_recording(num_channels=n_ch, durations=[DURATION_S], seed=SEED)
    rec = rec.set_probe(probe, in_place=False)

    # Spike trains for each unit
    unit_dict: dict[str, np.ndarray] = {}
    for i, fr in enumerate(firing_rates):
        n_spikes = int(max(1, round(float(fr) * DURATION_S)))
        times = np.sort(RNG.uniform(0.01, DURATION_S - 0.01, n_spikes))
        unit_dict[str(i)] = (times * FS).astype(np.int64)

    sorting = sic.NumpySorting.from_unit_dict(unit_dict, sampling_frequency=FS)
    sorting.register_recording(rec)

    # Create analyzer and compute the pipeline needed for unit_locations
    analyzer = sic.SortingAnalyzer.create(sorting, rec, format="memory")
    analyzer.compute(["random_spikes", "templates", "unit_locations"])

    # Override unit_locations with our desired positions
    unit_locs_3d = np.column_stack([
        unit_locs_mea,
        np.zeros(len(unit_locs_mea)),
    ]).astype(np.float64)
    analyzer.get_extension("unit_locations").set_data("unit_locations", unit_locs_3d)

    return analyzer


# ---------------------------------------------------------------------------
# Step 3: Verify planted effects (quick check before saving)
# ---------------------------------------------------------------------------

def verify_planted_effects(cell_coords: np.ndarray, cell_labels: list[str],
                            cell_expression: np.ndarray,
                            unit_locs_st: np.ndarray,
                            firing_rates: np.ndarray) -> None:
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr

    gene_idx = {g: i for i, g in enumerate(ALL_GENES)}
    pvalb_idx = gene_idx["Pvalb"]
    slc_idx = gene_idx["Slc17a7"]

    tree = cKDTree(cell_coords)
    pvalb_means = []
    slc_means = []

    for uloc in unit_locs_st:
        nbrs = tree.query_ball_point(uloc, r=FOOTPRINT_RADIUS)
        if len(nbrs) == 0:
            pvalb_means.append(0.0)
            slc_means.append(0.0)
        else:
            pvalb_means.append(float(cell_expression[nbrs, pvalb_idx].mean()))
            slc_means.append(float(cell_expression[nbrs, slc_idx].mean()))

    rho_pvalb, p_pvalb = spearmanr(pvalb_means, firing_rates)
    rho_slc, p_slc = spearmanr(slc_means, firing_rates)

    print(f"  Pvalb correlation:   ρ = {rho_pvalb:.3f}  p = {p_pvalb:.4f}")
    print(f"  Slc17a7 correlation: ρ = {rho_slc:.3f}  p = {p_slc:.4f}")

    if abs(rho_pvalb) < 0.3:
        print("  WARNING: Pvalb correlation is weaker than expected (< 0.3).")
    if p_slc < 0.05:
        print("  WARNING: Slc17a7 correlation is significant — planted effect may be contaminated.")

    # Cell-type effect size — derive unit_types_full from UNIT_TYPES dict order
    from scipy.stats import kruskal
    unit_types_full = []
    for utype, cfg in UNIT_TYPES.items():
        unit_types_full.extend([utype] * cfg["n"])
    pyr_fr = firing_rates[[i for i, t in enumerate(unit_types_full) if t == "Pyramidal_dominated"]]
    pvalb_fr = firing_rates[[i for i, t in enumerate(unit_types_full) if t == "Pvalb_dominated"]]
    pooled_std = np.std(np.concatenate([pyr_fr, pvalb_fr]))
    d = (pvalb_fr.mean() - pyr_fr.mean()) / (pooled_std + 1e-9)
    stat, p_kw = kruskal(pyr_fr, pvalb_fr)
    print(f"  Pvalb vs Pyr effect size d = {d:.2f},  KW p = {p_kw:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Generating synthetic Slide-tags data…")
    adata = make_slide_tags()
    cell_coords = adata.obsm["spatial"]
    cell_labels = adata.obs["cell_type"].tolist()
    cell_expression = np.asarray(adata.X.todense())

    st_path = OUT_DIR / "synthetic_slide_tags.h5ad"
    adata.write_h5ad(st_path)
    print(f"  Saved: {st_path}")
    print(f"  Cells: {adata.n_obs}  Genes: {adata.n_vars}")
    print(f"  Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")

    print("\nGenerating MEA unit layout…")
    unit_locs_st, unit_locs_mea, unit_types = make_unit_layout()

    firing_rates = make_firing_rates(unit_types, cell_coords, cell_labels, unit_locs_st)
    print(f"  Units: {N_UNITS}  FR range: [{firing_rates.min():.1f}, {firing_rates.max():.1f}] Hz")

    print("\nVerifying planted effects…")
    verify_planted_effects(cell_coords, cell_labels, cell_expression,
                           unit_locs_st, firing_rates)

    print("\nBuilding SortingAnalyzer…")
    analyzer = make_mea_sorting_analyzer(unit_locs_mea, firing_rates)

    mea_path = OUT_DIR / "synthetic_mea"
    if mea_path.exists():
        import shutil
        shutil.rmtree(mea_path)
    analyzer.save_as(folder=str(mea_path), format="binary_folder")
    print(f"  Saved: {mea_path}")

    print("\nSaving ground-truth transform…")
    gt = _gt_transform()
    transform_path = OUT_DIR / "ground_truth_transform.json"
    transform_data = {
        "matrix": gt.params.tolist(),
        "rotation_deg": GT_ROTATION_DEG,
        "translation_um": GT_TRANSLATION.tolist(),
        "description": (
            "Ground-truth affine transform: maps Slide-tags (fixed) → MEA (moving). "
            "Apply to Slide-tags coords to get MEA coords. "
            "The inverse maps MEA → Slide-tags."
        ),
    }
    transform_path.write_text(json.dumps(transform_data, indent=2))
    print(f"  Saved: {transform_path}")

    print("\nVerifying load via app loaders…")
    sys.path.insert(0, str(Path(__file__).parent))
    from data.loaders import load_slide_tags, load_mea

    st_loaded = load_slide_tags(str(st_path), spatial_units="μm", cell_type_col="cell_type")
    print(f"  SlideTagsData OK: {st_loaded.coords.shape[0]} cells, units={st_loaded.spatial_units}")

    mea_loaded = load_mea(str(mea_path))
    print(f"  MEAData OK: {len(mea_loaded.unit_ids)} units, "
          f"FR range [{mea_loaded.firing_rates.min():.1f}, {mea_loaded.firing_rates.max():.1f}] Hz, "
          f"loc_source={mea_loaded.location_source}")

    print("\nDone.  All synthetic files in:", OUT_DIR)


if __name__ == "__main__":
    main()
