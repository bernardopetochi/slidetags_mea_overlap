"""
Spatial association: assign cells to unit footprints.

Public API
----------
compute_overlap(cell_coords, cell_labels, expression_matrix, gene_names,
                unit_coords, unit_ids, firing_rates, radius, mode)
    -> (list[UnitSummary], per_cell_units: dict[int, list])

summary_table(unit_summaries) -> pd.DataFrame
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class UnitSummary:
    """Per-unit spatial association result.

    Attributes
    ----------
    unit_id:
        Unit identifier (str or int, as stored in MEAData).
    x, y:
        Unit location in Slide-tags coordinate space (after registration).
    firing_rate:
        Mean firing rate in Hz.
    n_cells_within:
        Number of cells within the footprint (un-weighted).
    cell_type_fractions:
        Dict mapping cell-type label → weighted fraction of cells.
        Empty dict if no cell types are available or no cells are nearby.
    mean_expression:
        Dict mapping gene name → weighted mean expression in nearby cells.
    total_weight:
        Sum of weights (equals n_cells_within for hard mode, or sum of
        Gaussian weights for gaussian mode).
    """
    unit_id: str
    x: float
    y: float
    firing_rate: float
    n_cells_within: int
    cell_type_fractions: dict[str, float] = field(default_factory=dict)
    mean_expression: dict[str, float] = field(default_factory=dict)
    total_weight: float = 0.0


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_overlap(
    cell_coords: np.ndarray,
    unit_coords: np.ndarray,
    unit_ids: np.ndarray,
    firing_rates: np.ndarray,
    radius: float,
    mode: Literal["hard", "gaussian"] = "hard",
    cell_labels: list[str] | None = None,
    expression_matrix: np.ndarray | None = None,
    gene_names: list[str] | None = None,
) -> tuple[list[UnitSummary], dict[int, list[str]]]:
    """Compute spatial association between cells and unit footprints.

    Parameters
    ----------
    cell_coords:
        (n_cells, 2) float array of cell positions in Slide-tags space (μm).
    unit_coords:
        (n_units, 2) float array of unit positions in Slide-tags space
        (i.e. after registration).
    unit_ids:
        (n_units,) array of unit identifiers.
    firing_rates:
        (n_units,) float array of firing rates in Hz.
    radius:
        Footprint radius in μm.
        Hard mode: cells ≤ radius are in, others out.
        Gaussian mode: σ = radius; weights = exp(−d²/2σ²).
    mode:
        ``"hard"`` or ``"gaussian"``.
    cell_labels:
        Optional (n_cells,) list of cell-type labels.  If None, cell-type
        fractions are not computed.
    expression_matrix:
        Optional (n_cells, n_genes) numeric array (dense or sparse).
        If None, mean_expression is not computed.
    gene_names:
        Gene names aligned to columns of expression_matrix.

    Returns
    -------
    unit_summaries:
        One UnitSummary per unit.
    per_cell_units:
        Dict mapping cell index → list of unit_id strings for units whose
        footprint contains that cell.
    """
    cell_coords = np.asarray(cell_coords, dtype=np.float64)
    unit_coords = np.asarray(unit_coords, dtype=np.float64)
    firing_rates = np.asarray(firing_rates, dtype=np.float64)

    # Resolve expression matrix to dense ndarray if sparse
    if expression_matrix is not None:
        import scipy.sparse as sp
        if sp.issparse(expression_matrix):
            expression_matrix = np.asarray(expression_matrix.todense())
        else:
            expression_matrix = np.asarray(expression_matrix, dtype=np.float64)

    # Build KD-tree on cell positions for fast radius queries
    tree = cKDTree(cell_coords)

    unit_summaries: list[UnitSummary] = []
    per_cell_units: dict[int, list[str]] = {i: [] for i in range(len(cell_coords))}

    for u_idx, (uid, uloc, fr) in enumerate(zip(unit_ids, unit_coords, firing_rates)):
        uid_str = str(uid)

        # All cells within radius (for hard mode) or search radius (gaussian)
        neighbor_indices = tree.query_ball_point(uloc, r=radius)

        if len(neighbor_indices) == 0:
            unit_summaries.append(UnitSummary(
                unit_id=uid_str,
                x=float(uloc[0]),
                y=float(uloc[1]),
                firing_rate=float(fr),
                n_cells_within=0,
            ))
            continue

        neighbor_indices = np.array(neighbor_indices, dtype=int)
        dists = np.linalg.norm(cell_coords[neighbor_indices] - uloc, axis=1)

        if mode == "hard":
            weights = np.ones(len(neighbor_indices), dtype=np.float64)
        else:  # gaussian
            sigma = radius
            weights = np.exp(-(dists ** 2) / (2 * sigma ** 2))

        total_weight = float(weights.sum())

        # Per-cell unit membership
        for ci in neighbor_indices:
            per_cell_units[ci].append(uid_str)

        # Cell-type fractions
        ct_fractions: dict[str, float] = {}
        if cell_labels is not None and total_weight > 0:
            for ci, w in zip(neighbor_indices, weights):
                ct = cell_labels[ci]
                ct_fractions[ct] = ct_fractions.get(ct, 0.0) + float(w)
            # Normalise to fractions
            ct_fractions = {ct: v / total_weight for ct, v in ct_fractions.items()}

        # Weighted mean expression
        mean_expr: dict[str, float] = {}
        if expression_matrix is not None and gene_names is not None and total_weight > 0:
            expr_subset = expression_matrix[neighbor_indices]  # (k, n_genes)
            weighted_mean = (weights[:, None] * expr_subset).sum(axis=0) / total_weight
            mean_expr = {g: float(weighted_mean[j]) for j, g in enumerate(gene_names)}

        unit_summaries.append(UnitSummary(
            unit_id=uid_str,
            x=float(uloc[0]),
            y=float(uloc[1]),
            firing_rate=float(fr),
            n_cells_within=len(neighbor_indices),
            cell_type_fractions=ct_fractions,
            mean_expression=mean_expr,
            total_weight=total_weight,
        ))

    # Remove cells with no associated units from dict (keep only non-empty)
    per_cell_units = {k: v for k, v in per_cell_units.items() if v}

    return unit_summaries, per_cell_units


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def summary_table(unit_summaries: list[UnitSummary]) -> pd.DataFrame:
    """Build a tidy DataFrame with one row per unit.

    Columns:
        unit_id, x, y, firing_rate, n_cells_within, total_weight,
        + one column per cell type (fraction_<CellType>),
        + one column per gene (expr_<Gene>).
    """
    rows = []
    for us in unit_summaries:
        row: dict = {
            "unit_id": us.unit_id,
            "x": us.x,
            "y": us.y,
            "firing_rate": us.firing_rate,
            "n_cells_within": us.n_cells_within,
            "total_weight": us.total_weight,
        }
        for ct, frac in us.cell_type_fractions.items():
            row[f"fraction_{ct}"] = frac
        for gene, val in us.mean_expression.items():
            row[f"expr_{gene}"] = val
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Fill missing fraction/expr columns with 0
    for col in df.columns:
        if col.startswith("fraction_") or col.startswith("expr_"):
            df[col] = df[col].fillna(0.0)
    return df
