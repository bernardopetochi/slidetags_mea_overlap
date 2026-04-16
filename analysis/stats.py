"""
Statistical analyses for the Slide-tags × MEA overlap tool.

Public API
----------
gene_firing_correlation(unit_summaries, gene_name, log_transform=False)
    -> GeneCorrelationResult

celltype_firing_comparison(unit_summaries, min_fraction=0.6)
    -> CelltypeComparisonResult
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, kruskal


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class GeneCorrelationResult:
    """Result of a gene ↔ firing-rate Spearman correlation.

    Attributes
    ----------
    gene:
        Gene name.
    rho:
        Spearman correlation coefficient.
    p_value:
        Two-sided p-value.
    n_units:
        Number of units with non-zero cells (units included in the test).
    x_values:
        Firing rates used (possibly log-transformed).
    y_values:
        Weighted mean expression values.
    log_transformed:
        Whether x_values were log₁₀-transformed.
    """
    gene: str
    rho: float
    p_value: float
    n_units: int
    x_values: np.ndarray   # firing rate (or log10 FR)
    y_values: np.ndarray   # mean expression
    log_transformed: bool


@dataclass
class CelltypeComparisonResult:
    """Result of cell-type firing-rate group comparison.

    Attributes
    ----------
    groups:
        Dict mapping dominant cell-type label → array of firing rates.
    dominant_fractions:
        Array of max cell-type fraction across all units (for reliability warning).
    unreliable:
        True if median dominant fraction < min_fraction.
    kw_statistic:
        Kruskal-Wallis H statistic.
    kw_p_value:
        Kruskal-Wallis p-value.
    pairwise:
        DataFrame with pairwise Mann-Whitney results (BH-corrected).
        Columns: group_a, group_b, U_statistic, p_raw, p_adj, significant.
    n_units_assigned:
        Number of units assigned to a dominant cell type.
    n_units_total:
        Total number of units considered.
    min_fraction:
        The min_fraction threshold used.
    """
    groups: dict[str, np.ndarray]
    dominant_fractions: np.ndarray
    unreliable: bool
    kw_statistic: float
    kw_p_value: float
    pairwise: pd.DataFrame
    n_units_assigned: int
    n_units_total: int
    min_fraction: float


# ---------------------------------------------------------------------------
# Gene – firing correlation
# ---------------------------------------------------------------------------

def gene_firing_correlation(
    unit_summaries: list,  # list[UnitSummary]
    gene_name: str,
    log_transform: bool = False,
) -> GeneCorrelationResult:
    """Compute Spearman correlation between per-unit weighted mean expression
    and firing rate.

    Only units that have ≥1 nearby cell (n_cells_within > 0) and a non-NaN
    expression value for the gene are included.

    Parameters
    ----------
    unit_summaries:
        Output of association.compute_overlap().
    gene_name:
        Gene to test.
    log_transform:
        If True, apply log₁₀(FR + 1e-6) before correlation.

    Returns
    -------
    GeneCorrelationResult
    """
    fr_list = []
    expr_list = []

    for us in unit_summaries:
        if us.n_cells_within == 0:
            continue
        expr_val = us.mean_expression.get(gene_name)
        if expr_val is None or np.isnan(expr_val):
            continue
        fr_list.append(us.firing_rate)
        expr_list.append(expr_val)

    fr_arr = np.array(fr_list, dtype=np.float64)
    expr_arr = np.array(expr_list, dtype=np.float64)

    if log_transform:
        fr_arr = np.log10(fr_arr + 1e-6)

    if len(fr_arr) < 3:
        return GeneCorrelationResult(
            gene=gene_name,
            rho=float("nan"),
            p_value=float("nan"),
            n_units=len(fr_arr),
            x_values=fr_arr,
            y_values=expr_arr,
            log_transformed=log_transform,
        )

    rho, p_val = spearmanr(fr_arr, expr_arr)

    return GeneCorrelationResult(
        gene=gene_name,
        rho=float(rho),
        p_value=float(p_val),
        n_units=len(fr_arr),
        x_values=fr_arr,
        y_values=expr_arr,
        log_transformed=log_transform,
    )


# ---------------------------------------------------------------------------
# Cell-type firing comparison
# ---------------------------------------------------------------------------

def celltype_firing_comparison(
    unit_summaries: list,  # list[UnitSummary]
    min_fraction: float = 0.6,
) -> CelltypeComparisonResult:
    """Group units by dominant cell type and compare firing rates.

    A unit is assigned a dominant cell type when its top cell-type fraction
    exceeds ``min_fraction``.  Units below the threshold are labelled
    "Ambiguous" and excluded from statistical tests.

    Parameters
    ----------
    unit_summaries:
        Output of association.compute_overlap().
    min_fraction:
        Minimum fraction for dominant-type assignment.

    Returns
    -------
    CelltypeComparisonResult
    """
    assignments: list[tuple[str, float]] = []  # (dominant_type, firing_rate)
    dom_fracs: list[float] = []

    for us in unit_summaries:
        if not us.cell_type_fractions:
            dom_fracs.append(0.0)
            continue
        top_ct = max(us.cell_type_fractions, key=us.cell_type_fractions.get)
        top_frac = us.cell_type_fractions[top_ct]
        dom_fracs.append(top_frac)
        if top_frac >= min_fraction:
            assignments.append((top_ct, us.firing_rate))

    dom_fracs_arr = np.array(dom_fracs)
    unreliable = float(np.median(dom_fracs_arr)) < min_fraction if len(dom_fracs_arr) > 0 else True

    # Group firing rates
    groups: dict[str, list[float]] = {}
    for ct, fr in assignments:
        groups.setdefault(ct, []).append(fr)
    groups_arr: dict[str, np.ndarray] = {ct: np.array(v) for ct, v in groups.items()}

    n_assigned = len(assignments)
    n_total = len(unit_summaries)

    # Kruskal-Wallis across all groups with ≥2 units
    valid_groups = [v for v in groups_arr.values() if len(v) >= 2]
    if len(valid_groups) >= 2:
        kw_stat, kw_p = kruskal(*valid_groups)
    else:
        kw_stat, kw_p = float("nan"), float("nan")

    # Pairwise Mann-Whitney with BH correction
    pairwise_df = _pairwise_mann_whitney(groups_arr)

    return CelltypeComparisonResult(
        groups=groups_arr,
        dominant_fractions=dom_fracs_arr,
        unreliable=unreliable,
        kw_statistic=float(kw_stat),
        kw_p_value=float(kw_p),
        pairwise=pairwise_df,
        n_units_assigned=n_assigned,
        n_units_total=n_total,
        min_fraction=min_fraction,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pairwise_mann_whitney(
    groups: dict[str, np.ndarray],
) -> pd.DataFrame:
    """All pairwise Mann-Whitney U tests with Benjamini-Hochberg correction."""
    from itertools import combinations

    group_names = [g for g, v in groups.items() if len(v) >= 2]
    rows = []

    for a, b in combinations(group_names, 2):
        u_stat, p_raw = mannwhitneyu(groups[a], groups[b], alternative="two-sided")
        rows.append({"group_a": a, "group_b": b, "U_statistic": float(u_stat),
                     "p_raw": float(p_raw)})

    if not rows:
        return pd.DataFrame(columns=["group_a", "group_b", "U_statistic",
                                     "p_raw", "p_adj", "significant"])

    df = pd.DataFrame(rows)
    df["p_adj"] = _bh_correction(df["p_raw"].values)
    df["significant"] = df["p_adj"] < 0.05
    return df


def _bh_correction(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Returns adjusted p-values (not just rejected mask) so callers can apply
    any alpha threshold they wish.
    """
    n = len(p_values)
    if n == 0:
        return np.array([])

    order = np.argsort(p_values)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)

    # BH adjusted p-value: p_adj[i] = min over j≥i of (n/j * p[order[j]])
    sorted_p = p_values[order]
    adjusted = np.minimum.accumulate((n / np.arange(1, n + 1) * sorted_p)[::-1])[::-1]
    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original order
    result = np.empty(n)
    result[order] = adjusted
    return result
