"""Progress tracker custom Streamlit component."""

import os
import streamlit.components.v1 as components

_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend")

_component_func = components.declare_component("progress_tracker", path=_COMPONENT_DIR)

MODULE_NAMES = [
    "EEG Fundamentals",
    "DL Primer",
    "Detection Challenge",
    "Multi-Scale Encoding",
    "Attention & Gating",
    "Learning Representations",
    "Foundation Models",
    "Results & Impact",
]

def progress_tracker(
    current: int,
    visited: list[int] | None = None,
    key: str | None = None,
    current_color: str | None = None,
):
    """Render the module progress tracker.

    Args:
        current: 1-indexed current module number (1-8).
        visited: List of 1-indexed module numbers the user has visited.
        key: Streamlit component key.
        current_color: Optional hex color for the "current" step highlight
            (e.g. the active paper's color). Defaults to amber.
    """
    if visited is None:
        visited = []
    return _component_func(
        current=current,
        visited=visited,
        modules=MODULE_NAMES,
        current_color=current_color or "#F59E0B",
        key=key,
        default=None,
    )
