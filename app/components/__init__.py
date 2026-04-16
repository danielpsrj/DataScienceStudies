"""
Reusable Streamlit components for the Data Science Platform.

This module provides standardized components for building consistent
concept pages with interactive examples, theory sections, code snippets,
and more.
"""

from .theory import theory_section
from .math import math_equation
from .demo import interactive_demo
from .code import code_tabs
from .applications import (
    display_applications,
    get_regression_applications,
    get_clustering_applications,
    get_time_series_applications,
)
from .pitfalls import (
    display_pitfalls,
    get_regression_pitfalls,
    get_clustering_pitfalls,
    get_classification_pitfalls,
)
from .references import (
    display_references,
    get_regression_references,
    get_clustering_references,
    get_general_data_science_references,
    search_references,
)

__all__ = [
    "theory_section",
    "math_equation",
    "interactive_demo",
    "code_tabs",
    "display_applications",
    "get_regression_applications",
    "get_clustering_applications",
    "get_time_series_applications",
    "display_pitfalls",
    "get_regression_pitfalls",
    "get_clustering_pitfalls",
    "get_classification_pitfalls",
    "display_references",
    "get_regression_references",
    "get_clustering_references",
    "get_general_data_science_references",
    "search_references",
]
