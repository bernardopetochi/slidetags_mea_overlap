"""
Dash layout for the Slide-tags × MEA Spatial Overlap Tool.

Panel structure
---------------
┌─────────────────────────────────────────────────────────────────────┐
│  Header                                                              │
├──────────────────────┬──────────────────────────────────────────────┤
│  Sidebar (Load +     │  Main content                                │
│  Settings)           │  ┌──────────────────────────────────────┐   │
│                      │  │  Tab: Visualize (side-by-side/overlay)│   │
│                      │  ├──────────────────────────────────────┤   │
│                      │  │  Tab: Register                        │   │
│                      │  ├──────────────────────────────────────┤   │
│                      │  │  Tab: Overlap & Stats                 │   │
│                      │  ├──────────────────────────────────────┤   │
│                      │  │  Tab: Export                          │   │
│                      │  └──────────────────────────────────────┘   │
└──────────────────────┴──────────────────────────────────────────────┘
"""

from __future__ import annotations

import dash_bootstrap_components as dbc
from dash import dcc, html


# ---------------------------------------------------------------------------
# Small reusable helpers
# ---------------------------------------------------------------------------

def _label(text: str) -> html.Label:
    return html.Label(text, className="fw-semibold mb-1")


def _section(title: str, *children) -> html.Div:
    return html.Div([
        html.H6(title, className="text-uppercase text-muted small mt-3 mb-2"),
        *children,
    ])


# ---------------------------------------------------------------------------
# Sidebar: load + settings
# ---------------------------------------------------------------------------

def build_sidebar() -> dbc.Col:
    return dbc.Col(
        width=3,
        className="bg-light border-end p-3 vh-100 overflow-auto",
        children=[
            html.H5("Slide-tags × MEA", className="fw-bold mb-0"),
            html.Small("Spatial Overlap Tool v1", className="text-muted"),
            html.Hr(),

            # ---- Slide-tags ----
            _section("Slide-tags (.h5ad)"),
            _label("File path"),
            dbc.Input(
                id="input-slidetags-path",
                placeholder="/path/to/data.h5ad",
                type="text",
                className="mb-2",
            ),
            _label("Spatial units"),
            dcc.Dropdown(
                id="dropdown-spatial-units",
                options=[
                    {"label": "μm", "value": "μm"},
                    {"label": "pixels", "value": "pixels"},
                ],
                value="μm",
                clearable=False,
                className="mb-2",
            ),
            dbc.Collapse(
                id="collapse-pixel-scale",
                is_open=False,
                children=[
                    _label("Scale factor (μm/pixel)"),
                    dbc.Input(
                        id="input-pixel-scale",
                        type="number",
                        value=1.0,
                        min=0.001,
                        step=0.001,
                        className="mb-2",
                    ),
                ],
            ),
            _label("Cell-type obs column (optional)"),
            dbc.Input(
                id="input-celltype-col",
                placeholder="cell_type",
                type="text",
                value="cell_type",
                className="mb-2",
            ),

            # ---- MEA ----
            _section("MEA (SpikeInterface folder)"),
            _label("Folder path"),
            dbc.Input(
                id="input-mea-path",
                placeholder="/path/to/sorting_analyzer",
                type="text",
                className="mb-2",
            ),

            # ---- Load button ----
            html.Hr(),
            dbc.Button(
                "Load datasets",
                id="btn-load",
                color="primary",
                className="w-100",
            ),
            dbc.Spinner(
                html.Div(id="load-status", className="mt-2 small"),
                color="primary",
                size="sm",
            ),

            # ---- Status badges ----
            html.Div(id="load-badges", className="mt-2"),
        ],
    )


# ---------------------------------------------------------------------------
# Visualize tab
# ---------------------------------------------------------------------------

def build_visualize_tab() -> dcc.Tab:
    controls = dbc.Row([
        dbc.Col([
            _label("Slide-tags color by"),
            dcc.Dropdown(
                id="dropdown-st-color",
                options=[{"label": "Cell type", "value": "__cell_type__"}],
                value="__cell_type__",
                clearable=False,
            ),
        ], width=4),
        dbc.Col([
            _label("Gene (for expression color)"),
            dbc.Input(
                id="input-gene-viz",
                placeholder="e.g. Pvalb",
                type="text",
            ),
        ], width=4),
        dbc.Col([
            _label("MEA color by"),
            dcc.Dropdown(
                id="dropdown-mea-color",
                options=[{"label": "Firing rate", "value": "firing_rate"}],
                value="firing_rate",
                clearable=False,
            ),
        ], width=4),
    ], className="mb-3 g-2")

    plots = dbc.Row([
        dbc.Col(
            dcc.Graph(
                id="graph-slidetags",
                config={"scrollZoom": True, "displayModeBar": True},
                style={"height": "520px"},
            ),
            width=6,
        ),
        dbc.Col(
            dcc.Graph(
                id="graph-mea",
                config={"scrollZoom": True, "displayModeBar": True},
                style={"height": "520px"},
            ),
            width=6,
        ),
    ])

    overlay_row = dbc.Row([
        dbc.Col([
            dbc.Switch(
                id="switch-overlay",
                label="Show overlay view (requires registration)",
                value=False,
                className="mb-2",
            ),
            dbc.Collapse(
                id="collapse-overlay",
                is_open=False,
                children=[
                    _label("Opacity"),
                    dcc.Slider(
                        id="slider-opacity",
                        min=0.1,
                        max=1.0,
                        step=0.05,
                        value=0.7,
                        marks={0.1: "0.1", 0.5: "0.5", 1.0: "1.0"},
                    ),
                    dcc.Graph(
                        id="graph-overlay",
                        config={"scrollZoom": True},
                        style={"height": "520px"},
                    ),
                ],
            ),
        ]),
    ], className="mt-3")

    return dcc.Tab(
        label="Visualize",
        value="tab-viz",
        children=html.Div([controls, plots, overlay_row], className="p-3"),
    )


# ---------------------------------------------------------------------------
# Register tab
# ---------------------------------------------------------------------------

def build_register_tab() -> dcc.Tab:
    instructions = dbc.Alert([
        html.Strong("How to register: "),
        "Click a landmark in the Slide-tags plot (left), then the corresponding "
        "point in the MEA plot (right).  Repeat for ≥3 pairs.  "
        "The transform is computed automatically once you have ≥3 pairs.",
    ], color="info", className="mb-3")

    landmark_controls = dbc.Row([
        dbc.Col([
            dbc.Button("Undo last landmark pair", id="btn-undo-landmark",
                       color="secondary", size="sm", className="me-2"),
            dbc.Button("Clear all landmarks", id="btn-clear-landmarks",
                       color="danger", outline=True, size="sm"),
        ]),
        dbc.Col([
            dbc.Switch(id="switch-moving-frame",
                       label="Transform Slide-tags instead of MEA",
                       value=False),
        ]),
    ], className="mb-3")

    plots = dbc.Row([
        dbc.Col(
            dcc.Graph(
                id="graph-reg-st",
                config={"scrollZoom": True},
                style={"height": "500px"},
            ),
            width=6,
        ),
        dbc.Col(
            dcc.Graph(
                id="graph-reg-mea",
                config={"scrollZoom": True},
                style={"height": "500px"},
            ),
            width=6,
        ),
    ])

    stats_panel = dbc.Card([
        dbc.CardHeader("Registration quality"),
        dbc.CardBody([
            html.Div(id="reg-stats-display", children="No transform computed yet."),
            dbc.Button("Compute / update transform", id="btn-compute-transform",
                       color="success", className="mt-2 me-2"),
            dbc.Button("Save transform JSON", id="btn-save-transform",
                       color="secondary", outline=True, className="mt-2 me-2"),
            html.Hr(className="my-3"),
            html.P("Load transform from file:", className="mb-1 small fw-semibold"),
            dbc.Row([
                dbc.Col(
                    dbc.Input(id="input-transform-path",
                              placeholder="/path/to/transform.json",
                              type="text", size="sm"),
                    width=8,
                ),
                dbc.Col(
                    dbc.Button("Load from path", id="btn-load-transform-path",
                               color="secondary", outline=True, size="sm",
                               className="w-100"),
                    width=4,
                ),
            ], className="g-2 mb-2"),
            dcc.Upload(
                id="upload-transform",
                children=dbc.Button("…or upload transform JSON",
                                    color="secondary", outline=True,
                                    size="sm"),
            ),
            dcc.Download(id="download-transform"),
        ]),
    ], className="mt-3")

    # Hidden store for landmarks
    landmark_store = dcc.Store(id="store-landmarks", data={"st": [], "mea": []})

    return dcc.Tab(
        label="Register",
        value="tab-reg",
        children=html.Div([
            instructions,
            landmark_controls,
            plots,
            stats_panel,
            landmark_store,
        ], className="p-3"),
    )


# ---------------------------------------------------------------------------
# Overlap & Stats tab
# ---------------------------------------------------------------------------

def build_stats_tab() -> dcc.Tab:
    footprint_controls = dbc.Card([
        dbc.CardHeader("Footprint settings"),
        dbc.CardBody(dbc.Row([
            dbc.Col([
                _label("Radius (μm)"),
                dbc.Input(id="input-footprint-radius", type="number",
                          value=50, min=1, step=1),
            ], width=3),
            dbc.Col([
                _label("Footprint type"),
                dcc.Dropdown(
                    id="dropdown-footprint-type",
                    options=[
                        {"label": "Hard radius", "value": "hard"},
                        {"label": "Gaussian-weighted", "value": "gaussian"},
                    ],
                    value="hard",
                    clearable=False,
                ),
            ], width=3),
            dbc.Col([
                _label("Genes for expression summary"),
                dcc.Dropdown(
                    id="dropdown-genes",
                    placeholder="Type to search genes…",
                    multi=True,
                    options=[],
                ),
            ], width=6),
        ])),
    ], className="mb-3")

    compute_btn = dbc.Row([
        dbc.Col([
            dbc.Button("Compute overlap", id="btn-compute-overlap",
                       color="primary", className="me-2"),
            dbc.Spinner(html.Span(id="overlap-status"), size="sm"),
        ]),
    ], className="mb-3")

    summary_table = dbc.Card([
        dbc.CardHeader("Per-unit summary table"),
        dbc.CardBody(html.Div(id="div-summary-table")),
    ], className="mb-3")

    gene_corr = dbc.Card([
        dbc.CardHeader("Gene – firing rate correlation"),
        dbc.CardBody([
            dbc.Row([
                dbc.Col([
                    _label("Gene"),
                    dcc.Dropdown(id="dropdown-corr-gene", options=[],
                                 placeholder="Select a gene"),
                ], width=4),
                dbc.Col([
                    dbc.Switch(id="switch-log-fr",
                               label="Log₁₀-transform firing rate", value=False),
                ], width=4, className="d-flex align-items-end"),
                dbc.Col([
                    dbc.Button("Plot", id="btn-plot-corr", color="primary"),
                ], width=2, className="d-flex align-items-end"),
            ], className="mb-2"),
            dcc.Graph(id="graph-gene-corr", style={"height": "420px"}),
            html.Div(id="corr-stats-text", className="small mt-1"),
        ]),
    ], className="mb-3")

    celltype_comparison = dbc.Card([
        dbc.CardHeader("Cell-type firing rate comparison"),
        dbc.CardBody([
            dbc.Alert(id="celltype-reliability-warning", color="warning",
                      is_open=False),
            dbc.Button("Plot", id="btn-plot-celltype", color="primary",
                       className="mb-2"),
            dcc.Graph(id="graph-celltype-box", style={"height": "420px"}),
            html.Div(id="celltype-stats-text", className="small mt-1"),
        ]),
    ], className="mb-3")

    return dcc.Tab(
        label="Overlap & Stats",
        value="tab-stats",
        children=html.Div([
            footprint_controls,
            compute_btn,
            summary_table,
            gene_corr,
            celltype_comparison,
        ], className="p-3"),
    )


# ---------------------------------------------------------------------------
# Export tab
# ---------------------------------------------------------------------------

def build_export_tab() -> dcc.Tab:
    return dcc.Tab(
        label="Export",
        value="tab-export",
        children=html.Div([
            html.H6("Export options", className="mb-3"),
            dbc.Row([
                dbc.Col(
                    dbc.Button("Download summary CSV",
                               id="btn-export-csv", color="success",
                               className="w-100"),
                    width=4,
                ),
                dbc.Col(
                    dbc.Button("Download transform JSON",
                               id="btn-export-transform", color="secondary",
                               className="w-100"),
                    width=4,
                ),
                dbc.Col(
                    dbc.Button("Download current plot (PNG)",
                               id="btn-export-png", color="secondary",
                               className="w-100"),
                    width=4,
                ),
            ], className="g-2"),
            dcc.Download(id="download-csv"),
            dcc.Download(id="download-transform-export"),
        ], className="p-3"),
    )


# ---------------------------------------------------------------------------
# Root layout
# ---------------------------------------------------------------------------

def build_layout() -> html.Div:
    main_tabs = dcc.Tabs(
        id="main-tabs",
        value="tab-viz",
        children=[
            build_visualize_tab(),
            build_register_tab(),
            build_stats_tab(),
            build_export_tab(),
        ],
    )

    return html.Div([
        # Hidden data stores (shared state between callbacks)
        dcc.Store(id="store-slidetags", storage_type="memory"),
        dcc.Store(id="store-mea", storage_type="memory"),
        dcc.Store(id="store-registered-pair", storage_type="memory"),
        dcc.Store(id="store-overlap-results", storage_type="memory"),

        dbc.Row([
            build_sidebar(),
            dbc.Col(
                width=9,
                className="p-0",
                children=main_tabs,
            ),
        ], className="g-0"),
    ])
