"""
Mathematical equation component for concept pages.
"""
from typing import Optional, Dict
import streamlit as st


def math_equation(
    equation: str,
    variables: Optional[Dict[str, str]] = None,
    title: str = "Mathematical Formulation",
    icon: str = "🧮",
    expandable: bool = True,
) -> None:
    """
    Display mathematical equation with variable explanations.
    
    Args:
        equation: LaTeX equation string (without $$ delimiters)
        variables: Dictionary of {variable: description}
        title: Section title
        icon: Icon to display before title
        expandable: Whether to put in expander
    """
    if expandable:
        with st.expander(f"{icon} {title}", expanded=False):
            _display_equation(equation, variables)
    else:
        st.subheader(f"{icon} {title}")
        _display_equation(equation, variables)


def _display_equation(equation: str, variables: Optional[Dict[str, str]]) -> None:
    """Display equation and variables."""
    # Display equation
    st.latex(equation)
    
    # Display variable explanations
    if variables:
        st.markdown("**Where:**")
        
        # Create columns for better layout if many variables
        if len(variables) > 4:
            cols = st.columns(2)
            col_idx = 0
            
            for var, desc in variables.items():
                with cols[col_idx % 2]:
                    st.markdown(f"- ${var}$: {desc}")
                col_idx += 1
        else:
            for var, desc in variables.items():
                st.markdown(f"- ${var}$: {desc}")


def matrix_equation(
    matrix_name: str,
    elements: list[list[str]],
    description: Optional[str] = None,
) -> None:
    """
    Display a matrix equation.
    
    Args:
        matrix_name: Name of the matrix
        elements: 2D list of matrix elements as strings
        description: Optional description of the matrix
    """
    # Build matrix LaTeX
    rows = []
    for row in elements:
        rows.append(" & ".join(row))
    
    matrix_latex = r"\begin{bmatrix} " + r" \\ ".join(rows) + r" \end{bmatrix}"
    
    # Display
    st.latex(f"{matrix_name} = {matrix_latex}")
    
    if description:
        st.markdown(f"*{description}*")


def derivation_steps(
    steps: list[tuple[str, str]],
    title: str = "Derivation Steps",
    icon: str = "📝",
) -> None:
    """
    Display derivation steps with explanations.
    
    Args:
        steps: List of (step_latex, explanation) tuples
        title: Section title
        icon: Icon to display before title
    """
    st.subheader(f"{icon} {title}")
    
    for i, (step_latex, explanation) in enumerate(steps, 1):
        with st.container(border=True):
            st.markdown(f"**Step {i}**")
            st.latex(step_latex)
            st.markdown(explanation)


def probability_notation(
    notation: str,
    meaning: str,
    examples: Optional[list[str]] = None,
) -> None:
    """
    Display probability notation with explanation.
    
    Args:
        notation: Probability notation (e.g., P(X|Y))
        meaning: Meaning of the notation
        examples: Optional list of example interpretations
    """
    with st.container(border=True):
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.latex(notation)
        
        with col2:
            st.markdown(f"**Meaning:** {meaning}")
            
            if examples:
                st.markdown("**Examples:**")
                for example in examples:
                    st.markdown(f"- {example}")


# Example usage:
"""
# In a concept page:

from app.components import math_equation, matrix_equation, derivation_steps

# Basic equation
math_equation(
    equation=r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon",
    variables={
        "y": "Dependent variable",
        "\\beta_0": "Intercept",
        "\\beta_i": "Coefficients",
        "\\epsilon": "Error term"
    }
)

# Matrix equation
matrix_equation(
    matrix_name="X",
    elements=[
        ["1", "x_{11}", "x_{12}"],
        ["1", "x_{21}", "x_{22}"],
        ["\\vdots", "\\vdots", "\\vdots"],
        ["1", "x_{n1}", "x_{n2}"]
    ],
    description="Design matrix with intercept column"
)

# Derivation steps
derivation_steps([
    (r"\hat{\beta} = (X^T X)^{-1} X^T y", "Normal equation for OLS"),
    (r"RSS = \sum_{i=1}^n (y_i - \hat{y}_i)^2", "Residual sum of squares"),
])
"""