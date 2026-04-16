"""
Data loaders for Slide-tags (.h5ad) and MEA (SpikeInterface SortingAnalyzer).

API assumptions verified against SpikeInterface 0.104.1:
- sic.load_sorting_analyzer(folder) -> SortingAnalyzer
- analyzer.sorting  -> BaseSorting (attribute, not property)
- analyzer.get_extension("unit_locations") -> ext or None; ext.get_data() -> (n,3) array (x, y, z)
- analyzer.get_extension("quality_metrics")  -> ext or None; ext.get_data() -> pd.DataFrame with "firing_rate" col
- analyzer.sorting.count_num_spikes_per_unit() -> dict {unit_id: n_spikes}
- analyzer.sorting.get_total_duration() -> float (seconds)
- analyzer.get_channel_locations() -> (n_ch, 2) array
- analyzer.sorting.get_unit_ids() -> array of unit_ids
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Slide-tags loader
# ---------------------------------------------------------------------------

def load_slide_tags(
    path: str | Path,
    spatial_units: str = "μm",
    pixel_scale: float = 1.0,
    cell_type_col: Optional[str] = "cell_type",
) -> "SlideTagsData":  # noqa: F821  (imported at runtime below)
    """Load a Slide-tags AnnData file.

    Parameters
    ----------
    path:
        Path to the .h5ad file.
    spatial_units:
        Declared units of ``obsm['spatial']``.  Either ``"μm"`` or
        ``"pixels"``.  If ``"pixels"``, coordinates are multiplied by
        ``pixel_scale`` to convert to μm.
    pixel_scale:
        Micrometers per pixel.  Only used when ``spatial_units == "pixels"``.
    cell_type_col:
        Name of the ``obs`` column that contains cell-type labels.  Pass
        ``None`` to skip.  If the column does not exist, a warning is issued
        and cell-type analyses are silently disabled.

    Returns
    -------
    SlideTagsData
    """
    import anndata  # local import so the rest of the module loads without anndata
    from .schema import SlideTagsData

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Slide-tags file not found: {path}")

    adata = anndata.read_h5ad(path)

    if "spatial" not in adata.obsm:
        raise KeyError(
            "adata.obsm['spatial'] not found.  "
            "Expected a (n_cells, 2) coordinate array."
        )

    coords = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"adata.obsm['spatial'] has unexpected shape {coords.shape}; "
            "expected (n_cells, ≥2)."
        )
    coords = coords[:, :2]  # keep first two dims; ignore z if present

    if spatial_units == "pixels":
        if pixel_scale == 1.0:
            warnings.warn(
                "spatial_units='pixels' but pixel_scale=1.0 (no conversion). "
                "The 50 μm default footprint radius will be in pixels, which "
                "may not be physically meaningful.  Pass the correct μm/pixel "
                "scale factor.",
                UserWarning,
                stacklevel=2,
            )
        coords = coords * float(pixel_scale)
        effective_units = "μm (converted from pixels)"
    else:
        effective_units = spatial_units

    # Resolve cell-type column
    resolved_col: Optional[str] = None
    cell_types: Optional[pd.Series] = None

    if cell_type_col is not None:
        if cell_type_col in adata.obs.columns:
            resolved_col = cell_type_col
            cell_types = adata.obs[cell_type_col].copy()
        else:
            warnings.warn(
                f"cell_type_col='{cell_type_col}' not found in adata.obs "
                f"(columns: {list(adata.obs.columns)}).  "
                "Cell-type analyses will be disabled.",
                UserWarning,
                stacklevel=2,
            )

    return SlideTagsData(
        adata=adata,
        coords=coords,
        spatial_units=effective_units,
        cell_type_col=resolved_col,
        cell_types=cell_types,
    )


# ---------------------------------------------------------------------------
# MEA / SpikeInterface loader
# ---------------------------------------------------------------------------

def load_mea(path: str | Path) -> "MEAData":  # noqa: F821
    """Load a SpikeInterface SortingAnalyzer folder.

    Verified against SpikeInterface 0.104.1.

    Parameters
    ----------
    path:
        Path to the SortingAnalyzer folder (or legacy WaveformExtractor).

    Returns
    -------
    MEAData
    """
    import spikeinterface.core as sic
    from .schema import MEAData

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MEA folder not found: {path}")

    # load_sorting_analyzer handles both SortingAnalyzer and legacy formats
    analyzer = sic.load_sorting_analyzer(str(path), load_extensions=True)

    unit_ids = np.asarray(analyzer.sorting.get_unit_ids())
    n_units = len(unit_ids)

    # ------------------------------------------------------------------
    # Unit locations
    # ------------------------------------------------------------------
    loc_ext = analyzer.get_extension("unit_locations")
    if loc_ext is not None:
        # get_data() returns (n_units, 3): x, y, z  — keep x/y
        locs_raw = loc_ext.get_data()
        locations = np.asarray(locs_raw[:, :2], dtype=np.float32)
        location_source = "unit_locations_extension"
    else:
        # Fallback 1: try per-unit max-amplitude channel
        locations, location_source = _fallback_locations(analyzer, unit_ids)

    # ------------------------------------------------------------------
    # Firing rates
    # ------------------------------------------------------------------
    qm_ext = analyzer.get_extension("quality_metrics")
    quality_metrics_df: Optional[pd.DataFrame] = None
    firing_rate_source: str

    if qm_ext is not None:
        try:
            qm_df = qm_ext.get_data()
            quality_metrics_df = qm_df
            if "firing_rate" in qm_df.columns:
                # Align to unit_ids order (qm_df is indexed by unit_id)
                firing_rates = np.asarray(
                    qm_df.loc[unit_ids, "firing_rate"], dtype=np.float32
                )
                firing_rate_source = "quality_metrics_extension"
            else:
                firing_rates, firing_rate_source = _compute_firing_rates(
                    analyzer, unit_ids
                )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Could not read quality_metrics extension ({exc}); "
                "computing firing rates from spike trains.",
                UserWarning,
                stacklevel=2,
            )
            firing_rates, firing_rate_source = _compute_firing_rates(
                analyzer, unit_ids
            )
    else:
        firing_rates, firing_rate_source = _compute_firing_rates(
            analyzer, unit_ids
        )

    return MEAData(
        analyzer=analyzer,
        unit_ids=unit_ids,
        locations=locations,
        firing_rates=firing_rates,
        quality_metrics=quality_metrics_df,
        location_source=location_source,
        firing_rate_source=firing_rate_source,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _fallback_locations(
    analyzer: "sic.SortingAnalyzer",  # type: ignore[name-defined]
    unit_ids: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return (n_units, 2) locations from max-amplitude channel or probe."""
    try:
        channel_locs = analyzer.get_channel_locations()  # (n_ch, 2)
        # Try to get max-amplitude channel per unit from templates
        templates_ext = analyzer.get_extension("templates")
        if templates_ext is not None:
            templates = templates_ext.get_data()  # (n_units, n_samples, n_ch)
            peak_amps = np.abs(templates).max(axis=1)  # (n_units, n_ch)
            best_ch = np.argmax(peak_amps, axis=1)  # (n_units,)
            locations = channel_locs[best_ch].astype(np.float32)
            return locations, "max_amplitude_channel"
    except Exception:  # noqa: BLE001
        pass

    # Final fallback: centroid of all channels
    try:
        channel_locs = analyzer.get_channel_locations()
        centroid = channel_locs.mean(axis=0, keepdims=True).astype(np.float32)
        locations = np.tile(centroid, (len(unit_ids), 1))
        warnings.warn(
            "unit_locations extension not found and templates unavailable. "
            "Falling back to probe centroid for all units.  "
            "Run si.compute_unit_locations() on your SortingAnalyzer for "
            "accurate unit positions.",
            UserWarning,
            stacklevel=3,
        )
        return locations, "probe_channel"
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Cannot determine unit locations: unit_locations extension is "
            f"absent and channel locations are unavailable ({exc}).  "
            "Run si.compute_unit_locations() first."
        ) from exc


def _compute_firing_rates(
    analyzer: "sic.SortingAnalyzer",  # type: ignore[name-defined]
    unit_ids: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Compute firing rates as n_spikes / total_duration.

    When the associated recording is not loaded (common after save/reload),
    duration is derived from analyzer.get_num_samples() / sampling_frequency,
    which is always stored in the SortingAnalyzer metadata.
    """
    sorting = analyzer.sorting

    try:
        duration = sorting.get_total_duration()  # requires recording
    except AssertionError:
        # Recording not attached; derive duration from analyzer metadata
        duration = analyzer.get_num_samples() / analyzer.sampling_frequency

    # count_num_spikes_per_unit returns a dict {unit_id: count}
    spike_counts = sorting.count_num_spikes_per_unit()

    firing_rates = np.array(
        [spike_counts.get(uid, 0) / duration for uid in unit_ids],
        dtype=np.float32,
    )
    return firing_rates, "computed_from_spike_trains"
