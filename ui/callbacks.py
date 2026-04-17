"""
Dash callbacks for the Slide-tags × MEA Spatial Overlap Tool.

Covered here:
  - Sidebar: load datasets, toggle pixel-scale input
  - Visualize tab: Slide-tags scatter, MEA scatter, color options
  - Register tab: mirror plots with landmark overlays, landmark click capture
  - Overlap & Stats tab: compute overlap, gene correlation, cell-type boxplot
  - Export tab: CSV download, transform JSON download
"""

from __future__ import annotations

import io
import json
import traceback
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import (
    Input, Output, State, callback, ctx, dcc, html,
    no_update,
)
from dash.exceptions import PreventUpdate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_fig(msg: str = "No data loaded") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font={"size": 14, "color": "#888"},
    )
    fig.update_layout(
        xaxis={"visible": False}, yaxis={"visible": False},
        plot_bgcolor="#f8f9fa", paper_bgcolor="#f8f9fa",
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
    )
    return fig


def _scatter_layout(title: str) -> dict:
    return dict(
        title=title,
        margin={"l": 30, "r": 10, "t": 40, "b": 30},
        uirevision=title,
        hovermode="closest",
        plot_bgcolor="white",
        xaxis={"scaleanchor": "y", "constrain": "domain",
               "showgrid": False, "zeroline": False},
        yaxis={"showgrid": False, "zeroline": False},
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _serialize_slidetags(st) -> dict:
    cell_types = None
    if st.cell_types is not None:
        cell_types = st.cell_types.astype(str).tolist()
    return {
        "coords": st.coords.tolist(),
        "cell_ids": st.adata.obs_names.tolist(),
        "cell_types": cell_types,
        "cell_type_col": st.cell_type_col,
        "spatial_units": st.spatial_units,
        "var_names": st.adata.var_names.tolist(),
        "gene_cache": {},
    }


def _serialize_mea(mea) -> dict:
    qm = None
    if mea.quality_metrics is not None:
        qm = mea.quality_metrics.to_dict(orient="index")
    return {
        "unit_ids": [str(u) for u in mea.unit_ids.tolist()],
        "locations": mea.locations.tolist(),
        "firing_rates": mea.firing_rates.tolist(),
        "quality_metrics": qm,
        "location_source": mea.location_source,
        "firing_rate_source": mea.firing_rate_source,
    }


# ---------------------------------------------------------------------------
# Callback registration
# ---------------------------------------------------------------------------

def register_callbacks(app):  # noqa: C901 — large but each section is clear
    """Attach all Dash callbacks to the app."""

    # ------------------------------------------------------------------
    # Sidebar: pixel-scale visibility
    # ------------------------------------------------------------------

    @app.callback(
        Output("collapse-pixel-scale", "is_open"),
        Input("dropdown-spatial-units", "value"),
    )
    def toggle_pixel_scale(units: str) -> bool:
        return units == "pixels"

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------

    @app.callback(
        Output("store-slidetags", "data"),
        Output("store-mea", "data"),
        Output("load-status", "children"),
        Output("load-badges", "children"),
        Output("dropdown-genes", "options"),
        Output("dropdown-corr-gene", "options"),
        Input("btn-load", "n_clicks"),
        State("input-slidetags-path", "value"),
        State("input-mea-path", "value"),
        State("dropdown-spatial-units", "value"),
        State("input-pixel-scale", "value"),
        State("input-celltype-col", "value"),
        prevent_initial_call=True,
    )
    def load_datasets(n_clicks, st_path, mea_path,
                      spatial_units, pixel_scale, celltype_col):
        if not st_path or not mea_path:
            return (no_update, no_update,
                    dbc.Alert("Please fill in both file paths.", color="warning"),
                    no_update, no_update, no_update)

        from data.loaders import load_slide_tags, load_mea

        try:
            st = load_slide_tags(
                st_path,
                spatial_units=spatial_units or "μm",
                pixel_scale=float(pixel_scale or 1.0),
                cell_type_col=celltype_col.strip() if celltype_col else None,
            )
        except Exception as exc:
            return (no_update, no_update,
                    dbc.Alert([html.Strong("Slide-tags error: "), str(exc)],
                              color="danger"),
                    no_update, no_update, no_update)

        try:
            mea = load_mea(mea_path)
        except Exception as exc:
            return (no_update, no_update,
                    dbc.Alert([html.Strong("MEA error: "), str(exc)],
                              color="danger"),
                    no_update, no_update, no_update)

        st_store = _serialize_slidetags(st)
        mea_store = _serialize_mea(mea)

        gene_options = [{"label": g, "value": g}
                        for g in sorted(st.adata.var_names.tolist())]

        badges = [
            dbc.Badge(f"Slide-tags: {st.coords.shape[0]:,} cells",
                      color="success", className="me-1"),
            dbc.Badge(f"MEA: {len(mea.unit_ids)} units", color="primary"),
        ]
        if st.cell_type_col:
            badges.append(dbc.Badge(f"Cell types: {st.cell_types.nunique()}",
                                    color="info", className="ms-1"))

        status = dbc.Alert(
            f"Loaded OK — loc_source: {mea.location_source} | "
            f"fr_source: {mea.firing_rate_source}",
            color="success",
        )
        return st_store, mea_store, status, badges, gene_options, gene_options

    # ------------------------------------------------------------------
    # Visualize: Slide-tags scatter
    # ------------------------------------------------------------------

    @app.callback(
        Output("graph-slidetags", "figure"),
        Input("store-slidetags", "data"),
        Input("dropdown-st-color", "value"),
        Input("input-gene-viz", "value"),
    )
    def update_slidetags_plot(st_data, color_by, gene):
        if not st_data:
            return _empty_fig("Load a Slide-tags dataset")

        coords = np.array(st_data["coords"])
        df = pd.DataFrame({
            "x": coords[:, 0], "y": coords[:, 1],
            "cell_id": st_data["cell_ids"],
        })

        color_col = None
        colorscale = None

        if color_by == "__cell_type__" and st_data.get("cell_types"):
            df["cell_type"] = st_data["cell_types"]
            color_col = "cell_type"
        elif color_by == "__gene__" and gene and gene in st_data.get("gene_cache", {}):
            df["expression"] = st_data["gene_cache"][gene]
            color_col = "expression"
            colorscale = "Viridis"

        fig = px.scatter(
            df, x="x", y="y", color=color_col,
            color_continuous_scale=colorscale,
            hover_data={"cell_id": True, "x": ":.1f", "y": ":.1f"},
            labels={"x": "x (μm)", "y": "y (μm)"},
        )
        fig.update_traces(marker={"size": 3, "opacity": 0.7})
        fig.update_layout(**_scatter_layout("Slide-tags"))
        return fig

    # ------------------------------------------------------------------
    # Visualize: MEA scatter
    # ------------------------------------------------------------------

    @app.callback(
        Output("graph-mea", "figure"),
        Input("store-mea", "data"),
        Input("dropdown-mea-color", "value"),
    )
    def update_mea_plot(mea_data, color_by):
        if not mea_data:
            return _empty_fig("Load a MEA dataset")

        locs = np.array(mea_data["locations"])
        df = pd.DataFrame({
            "x": locs[:, 0], "y": locs[:, 1],
            "unit_id": [str(u) for u in mea_data["unit_ids"]],
            "firing_rate": mea_data["firing_rates"],
        })
        color_col = "firing_rate" if color_by == "firing_rate" else None

        fig = px.scatter(
            df, x="x", y="y", color=color_col,
            color_continuous_scale="Plasma",
            hover_data={"unit_id": True, "firing_rate": ":.2f",
                        "x": ":.1f", "y": ":.1f"},
            labels={"x": "x (μm)", "y": "y (μm)", "firing_rate": "FR (Hz)"},
        )
        fig.update_traces(marker={
            "size": 10, "symbol": "diamond", "opacity": 0.9,
            "line": {"width": 1, "color": "white"},
        })
        fig.update_layout(**_scatter_layout("MEA units"))
        return fig

    # ------------------------------------------------------------------
    # Visualize: color-by dropdown update when gene is typed
    # ------------------------------------------------------------------

    @app.callback(
        Output("dropdown-st-color", "options"),
        Input("input-gene-viz", "value"),
        State("store-slidetags", "data"),
    )
    def update_color_options(gene, st_data):
        base = [{"label": "Cell type", "value": "__cell_type__"}]
        if gene and st_data and gene in (st_data.get("var_names") or []):
            base.append({"label": f"Gene: {gene}", "value": "__gene__"})
        return base

    # ------------------------------------------------------------------
    # Register tab: mirror scatter plots with landmark overlays
    # ------------------------------------------------------------------

    @app.callback(
        Output("graph-reg-st", "figure"),
        Output("graph-reg-mea", "figure"),
        Input("store-slidetags", "data"),
        Input("store-mea", "data"),
        Input("store-landmarks", "data"),
        Input("store-registered-pair", "data"),
    )
    def update_registration_plots(st_data, mea_data, landmarks, reg_data):
        st_fig = _empty_fig("Load a Slide-tags dataset")
        mea_fig = _empty_fig("Load a MEA dataset")

        if st_data:
            coords = np.array(st_data["coords"])
            df_st = pd.DataFrame({
                "x": coords[:, 0], "y": coords[:, 1],
                "cell_id": st_data["cell_ids"],
            })
            color_col = None
            if st_data.get("cell_types"):
                df_st["cell_type"] = st_data["cell_types"]
                color_col = "cell_type"

            st_fig = px.scatter(df_st, x="x", y="y", color=color_col,
                                hover_data={"cell_id": True},
                                labels={"x": "x (μm)", "y": "y (μm)"})
            st_fig.update_traces(marker={"size": 3, "opacity": 0.5})
            st_fig.update_layout(**_scatter_layout(
                "Slide-tags — click landmarks"))

            if landmarks and landmarks.get("st"):
                lm = np.array(landmarks["st"])
                st_fig.add_trace(go.Scatter(
                    x=lm[:, 0], y=lm[:, 1],
                    mode="markers+text",
                    marker={"size": 12, "color": "red", "symbol": "cross"},
                    text=[str(i + 1) for i in range(len(lm))],
                    textposition="top center",
                    name="Landmarks", showlegend=False,
                ))

        if mea_data:
            # Use registered locations if available, else raw
            if reg_data and reg_data.get("mea_locations_registered"):
                locs = np.array(reg_data["mea_locations_registered"])
                title = "MEA (registered) — click landmarks"
            else:
                locs = np.array(mea_data["locations"])
                title = "MEA (raw) — click landmarks"

            df_mea = pd.DataFrame({
                "x": locs[:, 0], "y": locs[:, 1],
                "unit_id": [str(u) for u in mea_data["unit_ids"]],
                "firing_rate": mea_data["firing_rates"],
            })
            mea_fig = px.scatter(df_mea, x="x", y="y", color="firing_rate",
                                 color_continuous_scale="Plasma",
                                 hover_data={"unit_id": True, "firing_rate": ":.2f"},
                                 labels={"x": "x (μm)", "y": "y (μm)"})
            mea_fig.update_traces(marker={"size": 10, "symbol": "diamond",
                                          "opacity": 0.8})
            mea_fig.update_layout(**_scatter_layout(title))

            if landmarks and landmarks.get("mea"):
                lm = np.array(landmarks["mea"])
                mea_fig.add_trace(go.Scatter(
                    x=lm[:, 0], y=lm[:, 1],
                    mode="markers+text",
                    marker={"size": 12, "color": "red", "symbol": "cross"},
                    text=[str(i + 1) for i in range(len(lm))],
                    textposition="top center",
                    name="Landmarks", showlegend=False,
                ))

        return st_fig, mea_fig

    # ------------------------------------------------------------------
    # Landmark click capture
    # ------------------------------------------------------------------

    @app.callback(
        Output("store-landmarks", "data"),
        Input("graph-reg-st", "clickData"),
        Input("graph-reg-mea", "clickData"),
        Input("btn-undo-landmark", "n_clicks"),
        Input("btn-clear-landmarks", "n_clicks"),
        State("store-landmarks", "data"),
        prevent_initial_call=True,
    )
    def update_landmarks(st_click, mea_click, _undo, _clear, current):
        triggered = ctx.triggered_id
        current = current or {"st": [], "mea": []}

        if triggered == "btn-clear-landmarks":
            return {"st": [], "mea": []}
        if triggered == "btn-undo-landmark":
            if current["st"]:
                current["st"] = current["st"][:-1]
            if current["mea"]:
                current["mea"] = current["mea"][:-1]
            return current
        if triggered == "graph-reg-st" and st_click:
            pt = st_click["points"][0]
            current["st"].append([pt["x"], pt["y"]])
            return current
        if triggered == "graph-reg-mea" and mea_click:
            pt = mea_click["points"][0]
            current["mea"].append([pt["x"], pt["y"]])
            return current
        raise PreventUpdate

    # ------------------------------------------------------------------
    # Compute transform
    # ------------------------------------------------------------------

    @app.callback(
        Output("store-registered-pair", "data"),
        Output("reg-stats-display", "children"),
        Input("btn-compute-transform", "n_clicks"),
        State("store-landmarks", "data"),
        State("store-mea", "data"),
        State("switch-moving-frame", "value"),
        prevent_initial_call=True,
    )
    def compute_transform(n_clicks, landmarks, mea_data, flip_moving):
        if not landmarks:
            return no_update, dbc.Alert("No landmarks yet.", color="warning")

        st_pts = landmarks.get("st", [])
        mea_pts = landmarks.get("mea", [])
        n_pairs = min(len(st_pts), len(mea_pts))

        if n_pairs < 3:
            return no_update, dbc.Alert(
                f"Need ≥3 landmark pairs; have {n_pairs}.", color="warning")

        from analysis.registration import compute_affine_transform, apply_transform

        src = np.array(mea_pts[:n_pairs], dtype=np.float64)
        dst = np.array(st_pts[:n_pairs], dtype=np.float64)
        if flip_moving:
            src, dst = dst, src

        try:
            transform, rms, residuals = compute_affine_transform(src, dst)
        except Exception as exc:
            return no_update, dbc.Alert(str(exc), color="danger")

        # Apply transform to all MEA locations
        if mea_data:
            locs_raw = np.array(mea_data["locations"])
            locs_reg = apply_transform(locs_raw, transform).tolist()
        else:
            locs_reg = []

        reg_store = {
            "matrix": transform.params.tolist(),
            "mea_locations_registered": locs_reg,
            "rms_error": rms,
            "residuals": residuals.tolist(),
            "n_landmarks": n_pairs,
        }

        residual_rows = [
            html.Tr([html.Td(f"#{i+1}"), html.Td(f"{r:.2f} μm")])
            for i, r in enumerate(residuals)
        ]
        stats_ui = html.Div([
            dbc.Alert(
                [html.Strong(f"RMS error: {rms:.2f} μm  "),
                 f"({n_pairs} landmark pairs)"],
                color="success" if rms < 20 else "warning",
            ),
            html.Details([
                html.Summary("Per-landmark residuals"),
                html.Table([
                    html.Thead(html.Tr([html.Th("Pair"), html.Th("Residual")])),
                    html.Tbody(residual_rows),
                ], className="table table-sm table-striped"),
            ]),
        ])
        return reg_store, stats_ui

    # ------------------------------------------------------------------
    # Save transform JSON
    # ------------------------------------------------------------------

    @app.callback(
        Output("download-transform", "data"),
        Input("btn-save-transform", "n_clicks"),
        State("store-registered-pair", "data"),
        prevent_initial_call=True,
    )
    def save_transform_cb(n_clicks, reg_data):
        if not reg_data:
            raise PreventUpdate
        from skimage.transform import AffineTransform
        from analysis.registration import _make_transform_dict
        transform = AffineTransform(matrix=np.array(reg_data["matrix"]))
        payload = _make_transform_dict(
            transform,
            rms_error_um=reg_data.get("rms_error", 0.0),
            n_landmarks=reg_data.get("n_landmarks", 0),
        )
        return dcc.send_string(json.dumps(payload, indent=2),
                               filename="transform.json")

    # ------------------------------------------------------------------
    # Compute overlap
    # ------------------------------------------------------------------

    @app.callback(
        Output("store-overlap-results", "data"),
        Output("overlap-status", "children"),
        Output("div-summary-table", "children"),
        Input("btn-compute-overlap", "n_clicks"),
        State("store-slidetags", "data"),
        State("store-mea", "data"),
        State("store-registered-pair", "data"),
        State("input-footprint-radius", "value"),
        State("dropdown-footprint-type", "value"),
        State("dropdown-genes", "value"),
        State("input-slidetags-path", "value"),
        prevent_initial_call=True,
    )
    def compute_overlap_cb(n_clicks, st_data, mea_data, reg_data,
                           radius, mode, selected_genes, st_path):
        if not st_data or not mea_data:
            return no_update, "Load both datasets first.", no_update

        radius = float(radius or 50.0)
        mode = mode or "hard"
        selected_genes = selected_genes or []

        # Use registered MEA locations if available, else raw
        if reg_data and reg_data.get("mea_locations_registered"):
            unit_locs = np.array(reg_data["mea_locations_registered"],
                                 dtype=np.float32)
        else:
            unit_locs = np.array(mea_data["locations"], dtype=np.float32)

        cell_coords = np.array(st_data["coords"], dtype=np.float32)
        cell_labels = st_data.get("cell_types")
        unit_ids = mea_data["unit_ids"]
        firing_rates = np.array(mea_data["firing_rates"], dtype=np.float32)

        from analysis.association import compute_overlap, summary_table

        # Load expression for selected genes directly from the h5ad file.
        # The gene_cache in the store only holds genes loaded for the Visualize
        # tab and is not guaranteed to contain the genes selected here.
        expr_matrix = None
        gene_names_used: list[str] = []
        if selected_genes and st_path:
            try:
                import anndata
                import scipy.sparse as sp
                adata = anndata.read_h5ad(st_path)
                cols = []
                for gene in selected_genes:
                    if gene in adata.var_names:
                        idx = adata.var_names.get_loc(gene)
                        col = adata.X[:, idx]
                        if sp.issparse(col):
                            col = np.asarray(col.todense()).ravel()
                        else:
                            col = np.asarray(col).ravel()
                        cols.append(col.astype(np.float32))
                        gene_names_used.append(gene)
                if cols:
                    expr_matrix = np.column_stack(cols)
            except Exception as exc:
                # Non-fatal: compute overlap without expression
                import warnings
                warnings.warn(f"Could not load expression from {st_path}: {exc}")

        unit_summaries, per_cell = compute_overlap(
            cell_coords=cell_coords,
            unit_coords=unit_locs,
            unit_ids=unit_ids,
            firing_rates=firing_rates,
            radius=radius,
            mode=mode,
            cell_labels=cell_labels,
            expression_matrix=expr_matrix,
            gene_names=gene_names_used if gene_names_used else None,
        )

        df = summary_table(unit_summaries)

        # Serialise summaries for stats callbacks
        summaries_store = [
            {
                "unit_id": us.unit_id,
                "x": us.x,
                "y": us.y,
                "firing_rate": us.firing_rate,
                "n_cells_within": us.n_cells_within,
                "cell_type_fractions": us.cell_type_fractions,
                "mean_expression": us.mean_expression,
                "total_weight": us.total_weight,
            }
            for us in unit_summaries
        ]

        table_ui = _df_to_table(df.head(200))
        status_msg = (
            f"Overlap computed: {len(unit_summaries)} units | "
            f"radius={radius} μm | mode={mode}"
        )

        # Store overlap parameters so the correlation callback can recompute
        # expression on-the-fly for any gene without re-running full overlap.
        overlap_meta = {
            "summaries": summaries_store,
            "radius": radius,
            "mode": mode,
            "st_path": st_path or "",
            "cell_coords": st_data["coords"],          # list of [x,y]
            "cell_labels": cell_labels,
        }

        return overlap_meta, status_msg, table_ui

    # ------------------------------------------------------------------
    # Gene expression cache: extract expression for selected gene from h5ad
    # (triggered when user picks a gene in the Overlap tab or Visualize tab)
    # ------------------------------------------------------------------

    @app.callback(
        Output("store-slidetags", "data", allow_duplicate=True),
        Input("dropdown-corr-gene", "value"),
        State("store-slidetags", "data"),
        State("input-slidetags-path", "value"),
        prevent_initial_call=True,
    )
    def cache_gene_expression(gene, st_data, st_path):
        """Load a single gene's expression into the store cache."""
        if not gene or not st_data or not st_path:
            raise PreventUpdate
        if gene in st_data.get("gene_cache", {}):
            raise PreventUpdate

        try:
            import anndata
            adata = anndata.read_h5ad(st_path)
            if gene not in adata.var_names:
                raise PreventUpdate
            import scipy.sparse as sp
            idx = adata.var_names.get_loc(gene)
            col = adata.X[:, idx]
            if sp.issparse(col):
                col = np.asarray(col.todense()).ravel()
            else:
                col = np.asarray(col).ravel()
            st_data = dict(st_data)
            st_data["gene_cache"] = dict(st_data.get("gene_cache", {}))
            st_data["gene_cache"][gene] = col.tolist()
            return st_data
        except Exception:
            raise PreventUpdate

    # ------------------------------------------------------------------
    # Gene–firing correlation plot
    # ------------------------------------------------------------------

    @app.callback(
        Output("graph-gene-corr", "figure"),
        Output("corr-stats-text", "children"),
        Input("btn-plot-corr", "n_clicks"),
        State("store-overlap-results", "data"),
        State("dropdown-corr-gene", "value"),
        State("switch-log-fr", "value"),
        prevent_initial_call=True,
    )
    def plot_gene_corr(n_clicks, overlap_meta, gene, log_transform):
        if not overlap_meta or not gene:
            return _empty_fig("Compute overlap first, then select a gene."), ""

        from analysis.stats import gene_firing_correlation
        from analysis.association import compute_overlap as _compute_overlap

        summaries = [_dict_to_unit_summary(d)
                     for d in overlap_meta["summaries"]]

        # If the gene wasn't in the overlap computation's expression matrix,
        # recompute weighted mean expression for this gene on the fly.
        # This makes the correlation button fully independent of which genes
        # were selected at overlap-compute time.
        needs_expression = any(
            gene not in s.mean_expression for s in summaries
            if s.n_cells_within > 0
        )

        if needs_expression:
            st_path = overlap_meta.get("st_path", "")
            cell_coords_raw = overlap_meta.get("cell_coords")
            cell_labels = overlap_meta.get("cell_labels")
            radius = float(overlap_meta.get("radius", 50.0))
            mode = overlap_meta.get("mode", "hard")

            if not st_path or not cell_coords_raw:
                return _empty_fig(
                    f"Gene '{gene}' expression not in overlap results. "
                    "Re-compute overlap with this gene selected."
                ), ""

            try:
                import anndata, scipy.sparse as sp
                adata = anndata.read_h5ad(st_path)
                if gene not in adata.var_names:
                    return _empty_fig(f"Gene '{gene}' not found in dataset."), ""

                idx = adata.var_names.get_loc(gene)
                col = adata.X[:, idx]
                if sp.issparse(col):
                    col = np.asarray(col.todense()).ravel()
                else:
                    col = np.asarray(col).ravel()
                expr_col = col.astype(np.float32)[:, None]

                cell_coords = np.array(cell_coords_raw, dtype=np.float32)
                unit_coords = np.array([[s.x, s.y] for s in summaries],
                                       dtype=np.float32)
                unit_ids = np.array([s.unit_id for s in summaries])
                firing_rates = np.array([s.firing_rate for s in summaries],
                                        dtype=np.float32)

                fresh_summaries, _ = _compute_overlap(
                    cell_coords=cell_coords,
                    unit_coords=unit_coords,
                    unit_ids=unit_ids,
                    firing_rates=firing_rates,
                    radius=radius,
                    mode=mode,
                    cell_labels=cell_labels,
                    expression_matrix=expr_col,
                    gene_names=[gene],
                )
                # Merge fresh expression into existing summaries
                expr_by_unit = {s.unit_id: s.mean_expression.get(gene, 0.0)
                                for s in fresh_summaries}
                for s in summaries:
                    s.mean_expression[gene] = expr_by_unit.get(s.unit_id, 0.0)

            except Exception as exc:
                return _empty_fig(f"Error loading expression: {exc}"), ""

        result = gene_firing_correlation(summaries, gene,
                                         log_transform=bool(log_transform))

        if result.n_units < 3:
            return _empty_fig(
                f"Too few units with nearby cells for '{gene}' (n={result.n_units}). "
                "Try a larger footprint radius."
            ), ""

        x_label = ("log₁₀(Firing rate + ε) (Hz)" if log_transform
                   else "Firing rate (Hz)")
        df_plot = pd.DataFrame({
            x_label: result.x_values,
            f"Mean {gene} expression": result.y_values,
        })

        fig = px.scatter(
            df_plot, x=x_label, y=f"Mean {gene} expression",
            trendline="ols",
        )
        fig.update_traces(marker={"size": 8, "opacity": 0.7})
        fig.update_layout(
            title=f"{gene} vs firing rate",
            margin={"l": 40, "r": 10, "t": 50, "b": 40},
        )

        stats_text = (
            f"Spearman ρ = {result.rho:.3f}  |  "
            f"p = {result.p_value:.4f}  |  "
            f"n = {result.n_units} units"
        )
        significance = (
            " *** (p < 0.001)" if result.p_value < 0.001 else
            " ** (p < 0.01)"  if result.p_value < 0.01  else
            " * (p < 0.05)"   if result.p_value < 0.05  else
            " (not significant)"
        )

        return fig, stats_text + significance

    # ------------------------------------------------------------------
    # Cell-type firing comparison
    # ------------------------------------------------------------------

    @app.callback(
        Output("graph-celltype-box", "figure"),
        Output("celltype-stats-text", "children"),
        Output("celltype-reliability-warning", "is_open"),
        Output("celltype-reliability-warning", "children"),
        Input("btn-plot-celltype", "n_clicks"),
        State("store-overlap-results", "data"),
        prevent_initial_call=True,
    )
    def plot_celltype_comparison(n_clicks, overlap_meta):
        if not overlap_meta:
            return _empty_fig("Compute overlap first."), "", False, ""

        from analysis.stats import celltype_firing_comparison

        summaries = [_dict_to_unit_summary(d)
                     for d in overlap_meta["summaries"]]
        result = celltype_firing_comparison(summaries)

        if not result.groups:
            return (_empty_fig("No cell-type data — load dataset with cell types."),
                    "", False, "")

        # Build long-form dataframe for box plot
        rows = []
        for ct, frs in result.groups.items():
            for fr in frs:
                rows.append({"Cell type": ct, "Firing rate (Hz)": float(fr)})
        df = pd.DataFrame(rows)

        fig = px.box(df, x="Cell type", y="Firing rate (Hz)",
                     color="Cell type", points="all",
                     title="Firing rate by dominant cell type")
        fig.update_layout(margin={"l": 40, "r": 10, "t": 50, "b": 60},
                          showlegend=False)

        stats_lines = [
            f"Kruskal-Wallis H = {result.kw_statistic:.3f},  "
            f"p = {result.kw_p_value:.4f}  |  "
            f"{result.n_units_assigned}/{result.n_units_total} units assigned",
        ]
        if not result.pairwise.empty:
            sig_pairs = result.pairwise[result.pairwise["significant"]]
            if not sig_pairs.empty:
                pw_lines = [
                    f"  {r['group_a']} vs {r['group_b']}: "
                    f"p_adj = {r['p_adj']:.4f} *"
                    for _, r in sig_pairs.iterrows()
                ]
                stats_lines.extend(pw_lines)

        warn_msg = (
            f"Warning: median dominant-type fraction = "
            f"{np.median(result.dominant_fractions):.2f} < {result.min_fraction:.2f}.  "
            "Grouping may be unreliable — consider decreasing the footprint radius."
        )

        return (fig,
                html.Pre("\n".join(stats_lines), className="small"),
                result.unreliable,
                warn_msg)

    # ------------------------------------------------------------------
    # Export: CSV download
    # ------------------------------------------------------------------

    @app.callback(
        Output("download-csv", "data"),
        Input("btn-export-csv", "n_clicks"),
        State("store-overlap-results", "data"),
        prevent_initial_call=True,
    )
    def export_csv(n_clicks, overlap_meta):
        if not overlap_meta:
            raise PreventUpdate
        from analysis.association import summary_table
        summaries = [_dict_to_unit_summary(d)
                     for d in overlap_meta["summaries"]]
        df = summary_table(summaries)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        return dcc.send_string(buf.getvalue(), filename="overlap_summary.csv")

    # ------------------------------------------------------------------
    # Export: transform JSON
    # ------------------------------------------------------------------

    @app.callback(
        Output("download-transform-export", "data"),
        Input("btn-export-transform", "n_clicks"),
        State("store-registered-pair", "data"),
        prevent_initial_call=True,
    )
    def export_transform(n_clicks, reg_data):
        if not reg_data:
            raise PreventUpdate
        from skimage.transform import AffineTransform
        from analysis.registration import _make_transform_dict
        transform = AffineTransform(matrix=np.array(reg_data["matrix"]))
        payload = _make_transform_dict(
            transform,
            rms_error_um=reg_data.get("rms_error", 0.0),
            n_landmarks=reg_data.get("n_landmarks", 0),
        )
        return dcc.send_string(json.dumps(payload, indent=2),
                               filename="transform.json")

    # ------------------------------------------------------------------
    # Load transform from path or file upload
    # ------------------------------------------------------------------

    @app.callback(
        Output("store-registered-pair", "data", allow_duplicate=True),
        Output("reg-stats-display", "children", allow_duplicate=True),
        Input("btn-load-transform-path", "n_clicks"),
        State("input-transform-path", "value"),
        State("store-mea", "data"),
        prevent_initial_call=True,
    )
    def load_transform_from_path(n_clicks, path, mea_data):
        """Load a saved transform JSON from a file-system path."""
        if not path:
            raise PreventUpdate
        return _apply_transform_json_str(
            _read_transform_path(path), mea_data
        )

    @app.callback(
        Output("store-registered-pair", "data", allow_duplicate=True),
        Output("reg-stats-display", "children", allow_duplicate=True),
        Input("upload-transform", "contents"),
        State("upload-transform", "filename"),
        State("store-mea", "data"),
        prevent_initial_call=True,
    )
    def load_transform_from_upload(contents, filename, mea_data):
        """Load a saved transform JSON uploaded via the browser."""
        if not contents:
            raise PreventUpdate
        import base64
        # contents is "data:<mime>;base64,<data>"
        _, b64 = contents.split(",", 1)
        json_str = base64.b64decode(b64).decode("utf-8")
        return _apply_transform_json_str(json_str, mea_data)

    # ------------------------------------------------------------------
    # Overlay view toggle
    # ------------------------------------------------------------------

    @app.callback(
        Output("collapse-overlay", "is_open"),
        Input("switch-overlay", "value"),
    )
    def toggle_overlay(value: bool) -> bool:
        return bool(value)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _df_to_table(df: pd.DataFrame) -> Any:
    """Render a pandas DataFrame as a Dash Bootstrap table."""
    if df is None or df.empty:
        return html.P("No data.", className="text-muted")

    # Round floats for display
    df_disp = df.copy()
    for col in df_disp.select_dtypes(include="float"):
        df_disp[col] = df_disp[col].round(4)

    header = html.Thead(html.Tr([html.Th(c) for c in df_disp.columns]))
    rows = [
        html.Tr([html.Td(str(v)) for v in row])
        for row in df_disp.itertuples(index=False)
    ]
    return html.Div(
        html.Table(
            [header, html.Tbody(rows)],
            className="table table-sm table-striped table-bordered",
        ),
        style={"overflowX": "auto", "maxHeight": "400px", "overflowY": "auto"},
    )


def _dict_to_unit_summary(d: dict):
    """Reconstruct a UnitSummary from its JSON-serialised dict."""
    from analysis.association import UnitSummary
    return UnitSummary(
        unit_id=d["unit_id"],
        x=d["x"],
        y=d["y"],
        firing_rate=d["firing_rate"],
        n_cells_within=d["n_cells_within"],
        cell_type_fractions=d.get("cell_type_fractions", {}),
        mean_expression=d.get("mean_expression", {}),
        total_weight=d.get("total_weight", 0.0),
    )


def _read_transform_path(path: str) -> str:
    """Read a transform JSON file from a filesystem path, return as string."""
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Transform file not found: {path}")
    return p.read_text()


def _apply_transform_json_str(json_str: str, mea_data: dict | None):
    """
    Parse a transform JSON string, apply to MEA locations, return
    (reg_store dict, stats_ui component) for the Dash callbacks.

    Invariant: the "matrix" in the JSON must map MEA (moving) → Slide-tags (fixed).
    The matrix is applied directly to MEA unit coordinates — no inversion is performed.
    Use save_transform() (or the UI save button) to produce compliant JSON files.
    """
    import dash_bootstrap_components as dbc
    from dash import html
    from skimage.transform import AffineTransform
    from analysis.registration import apply_transform

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return no_update, dbc.Alert(f"Invalid JSON: {exc}", color="danger")

    if "matrix" not in data:
        return no_update, dbc.Alert(
            "JSON must contain a 'matrix' key (3×3 homogeneous affine).",
            color="danger",
        )

    matrix = np.array(data["matrix"])
    transform = AffineTransform(matrix=matrix)

    locs_reg: list = []
    if mea_data:
        locs_raw = np.array(mea_data["locations"])
        locs_reg = apply_transform(locs_raw, transform).tolist()

    rms = float(data.get("rms_error_um", data.get("rms_error", 0.0)))
    n_lm = int(data.get("n_landmarks", data.get("n_landmark_pairs", 0)))

    reg_store = {
        "matrix": matrix.tolist(),
        "mea_locations_registered": locs_reg,
        "rms_error": rms,
        "residuals": [],
        "n_landmarks": n_lm,
    }

    extra = []
    if data.get("rotation_deg") is not None:
        extra.append(f"rotation: {data['rotation_deg']}°")
    if data.get("translation_um") is not None:
        extra.append(f"translation: {data['translation_um']} μm")
    extra_str = "  |  " + ", ".join(extra) if extra else ""

    stats_ui = dbc.Alert(
        [
            html.Strong(
                f"Transform loaded from file.  "
                f"RMS error: {rms:.3f} μm"
                + (f"  ({n_lm} landmark pairs)" if n_lm else "") + extra_str
            ),
        ],
        color="success",
    )

    return reg_store, stats_ui
