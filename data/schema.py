"""
Dataclasses representing loaded datasets.

Assumptions (state explicitly so callers know):
- SlideTagsData.coords is always in the user's declared spatial units (μm or pixels);
  pixel→μm conversion is applied at load time if a scale factor is provided.
- MEAData.locations is always (n_units, 2) float32 in the MEA's native units (μm).
- Neither class owns the full expression matrix to avoid memory duplication;
  expression lookup is done on demand via SlideTagsData.adata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SlideTagsData:
    """Loaded Slide-tags dataset.

    Attributes
    ----------
    adata:
        The full AnnData object (cells × genes).
    coords:
        (n_cells, 2) float32 array of spatial coordinates after any unit
        conversion.  Row order matches adata.obs.
    spatial_units:
        Human-readable unit string, e.g. "μm" or "pixels".
    cell_type_col:
        Name of the obs column that holds cell-type labels, or None if absent.
    cell_types:
        Series of cell-type labels aligned to adata.obs, or None.
    """

    adata: Any  # anndata.AnnData
    coords: np.ndarray  # (n_cells, 2)
    spatial_units: str
    cell_type_col: str | None
    cell_types: pd.Series | None


@dataclass
class MEAData:
    """Loaded MEA / SpikeInterface dataset.

    Attributes
    ----------
    analyzer:
        The SortingAnalyzer object (or None if loaded from legacy format).
    unit_ids:
        1-D array of unit identifiers (str or int, as stored).
    locations:
        (n_units, 2) float32 array of unit spatial locations in μm.
        Derived from the ``unit_locations`` extension if available; otherwise
        falls back to the max-amplitude channel position.
    firing_rates:
        (n_units,) float32 array of mean firing rates in Hz.
        Derived from quality_metrics["firing_rate"] if available; otherwise
        computed from spike counts ÷ total recording duration.
    quality_metrics:
        DataFrame of quality metrics indexed by unit_id, or None if the
        extension was not computed.
    location_source:
        Which method was used to obtain locations: "unit_locations_extension",
        "max_amplitude_channel", or "probe_channel".
    firing_rate_source:
        "quality_metrics_extension" or "computed_from_spike_trains".
    """

    analyzer: Any  # spikeinterface SortingAnalyzer
    unit_ids: np.ndarray
    locations: np.ndarray  # (n_units, 2) float32
    firing_rates: np.ndarray  # (n_units,) float32
    quality_metrics: pd.DataFrame | None
    location_source: str
    firing_rate_source: str


@dataclass
class RegisteredPair:
    """A pair of datasets after affine registration.

    The Slide-tags dataset is always the fixed reference.
    The MEA locations are transformed into Slide-tags coordinate space.

    Attributes
    ----------
    slide_tags:
        The original (untransformed) SlideTagsData.
    mea:
        The original MEAData (untransformed locations stored here).
    mea_locations_registered:
        (n_units, 2) MEA locations after applying the affine transform.
    transform_matrix:
        3×3 homogeneous affine matrix (float64).
    landmarks_slide_tags:
        (n, 2) landmark coordinates clicked in the Slide-tags view.
    landmarks_mea:
        (n, 2) landmark coordinates clicked in the MEA view.
    rms_error:
        Root-mean-square residual in Slide-tags units after fitting.
    residuals:
        Per-landmark residual distances (n,).
    """

    slide_tags: SlideTagsData
    mea: MEAData
    mea_locations_registered: np.ndarray  # (n_units, 2)
    transform_matrix: np.ndarray  # (3, 3)
    landmarks_slide_tags: np.ndarray  # (n, 2)
    landmarks_mea: np.ndarray  # (n, 2)
    rms_error: float
    residuals: np.ndarray  # (n,)
