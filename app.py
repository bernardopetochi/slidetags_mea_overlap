"""
Entry point for the Slide-tags × MEA Spatial Overlap Tool.

Run with:
    uv run python app.py
or (after installing):
    slidetags-mea
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path when running as a script
sys.path.insert(0, str(Path(__file__).parent))

import dash
import dash_bootstrap_components as dbc

from ui.layout import build_layout
from ui.callbacks import register_callbacks


def create_app() -> dash.Dash:
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.BOOTSTRAP],
        title="Slide-tags × MEA Overlap",
        # suppress_callback_exceptions because some outputs are in collapsed
        # sections that may not be in the DOM at startup
        suppress_callback_exceptions=True,
    )
    app.layout = build_layout()
    register_callbacks(app)
    return app


def main():
    app = create_app()
    app.run(debug=True, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
