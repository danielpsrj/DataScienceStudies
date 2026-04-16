"""
Code display component for concept pages.
"""

from typing import Dict
import streamlit as st


def code_tabs(implementations: Dict[str, str]) -> None:
    """
    Display code implementations in tabs.

    Args:
        implementations: Dictionary of {tab_name: code_string}
    """
    if not implementations:
        st.warning("No code implementations provided")
        return

    tab_names = list(implementations.keys())
    tabs = st.tabs(tab_names)

    for tab, (name, code) in zip(tabs, implementations.items()):
        with tab:
            st.code(code, language="python")

            # Optional: Add download button
            if st.button(f"📋 Copy {name} Code", key=f"copy_{name}"):
                st.toast(f"Copied {name} code to clipboard!", icon="✅")
                # Note: Actual clipboard copy requires JavaScript
                # This is a visual indicator only


def code_snippet(
    code: str,
    language: str = "python",
    title: str = "Code Snippet",
    show_line_numbers: bool = True,
    copy_button: bool = True,
) -> None:
    """
    Display a code snippet with optional copy button.

    Args:
        code: Code string to display
        language: Programming language for syntax highlighting
        title: Optional title for the snippet
        show_line_numbers: Whether to show line numbers
        copy_button: Whether to show copy button
    """
    if title:
        st.subheader(title)

    st.code(code, language=language, line_numbers=show_line_numbers)

    if copy_button:
        if st.button("📋 Copy Code", key=f"copy_{hash(code)}"):
            st.toast("Code copied to clipboard!", icon="✅")


def code_comparison(
    implementations: Dict[str, str],
    title: str = "Implementation Comparison",
) -> None:
    """
    Display side-by-side code comparison.

    Args:
        implementations: Dictionary of {implementation_name: code}
        title: Section title
    """
    st.subheader(title)

    # Create columns for side-by-side comparison
    n_implementations = len(implementations)
    cols = st.columns(n_implementations)

    for col, (name, code) in zip(cols, implementations.items()):
        with col:
            st.markdown(f"**{name}**")
            st.code(code, language="python")

            # Show code stats
            lines = code.strip().split("\n")
            st.caption(f"{len(lines)} lines, {sum(len(line) for line in lines)} chars")


def algorithm_pseudocode(
    steps: list[str],
    title: str = "Algorithm Pseudocode",
    language: str = "python",
) -> None:
    """
    Display algorithm pseudocode.

    Args:
        steps: List of pseudocode steps
        title: Section title
        language: Language for syntax highlighting
    """
    st.subheader(title)

    pseudocode = "\n".join([f"{i + 1}. {step}" for i, step in enumerate(steps)])

    with st.container(border=True):
        st.code(pseudocode, language=language, line_numbers=False)

    # Optional: Add explanation
    with st.expander("📝 Step-by-step Explanation"):
        for i, step in enumerate(steps, 1):
            st.markdown(f"**Step {i}:** {step}")


# Example usage:
"""
# In a concept page:

from app.components import code_tabs, code_snippet, code_comparison

# Multiple implementations in tabs
code_tabs({
    "Scikit-learn": \"\"\"
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)
\"\"\",
    "NumPy": \"\"\"
import numpy as np
beta = np.linalg.inv(X.T @ X) @ X.T @ y
\"\"\",
})

# Single code snippet
code_snippet(
    code=\"\"\"
def calculate_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)
\"\"\",
    title="R² Calculation",
    language="python",
)

# Side-by-side comparison
code_comparison({
    "Python": \"\"\"
def sum_squares(n):
    return sum(i**2 for i in range(1, n+1))
\"\"\",
    "NumPy": \"\"\"
import numpy as np
def sum_squares(n):
    return np.sum(np.arange(1, n+1) ** 2)
\"\"\",
})
"""
