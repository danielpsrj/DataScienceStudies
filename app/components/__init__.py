"""
Reusable Streamlit components for the Data Science Platform.

This module provides standardized components for building consistent
concept pages with interactive examples, theory sections, code snippets,
and more.
"""

from .page_theory import theory_section
from .math import math_equation
from .demo import interactive_demo
from .code import code_tabs
from .applications import display_applications
from .page_pitfalls import display_pitfalls
from .page_references import display_references, search_references
from .demo_regression import regression_demo, get_regression_demo_code
from .demo_clustering import clustering_demo, get_clustering_demo_code

__all__ = [
    "theory_section",
    "math_equation",
    "interactive_demo",
    "code_tabs",
    "display_applications",
    "display_pitfalls",
    "display_references",
    "search_references",
    "regression_demo",
    "get_regression_demo_code",
    "clustering_demo",
    "get_clustering_demo_code",
]
