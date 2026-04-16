"""
Theory section component for concept pages.
"""

from typing import Optional, Tuple
import streamlit as st


def theory_section(
    title: str,
    content: str,
    image_url: Optional[str] = None,
    columns: Tuple[int, int] = (2, 1),
    expanded: bool = True,
) -> None:
    """
    Display a standardized theory section with optional image.

    Args:
        title: Section title (without emoji, will be prefixed with 📚)
        content: Markdown content explaining the concept
        image_url: Optional URL or path to an image
        columns: Tuple of column ratios (content, image)
        expanded: Whether the section should be expanded by default
    """
    if expanded:
        # Display directly without expander
        st.header(f"📚 {title}")

        if image_url:
            col1, col2 = st.columns(columns)
            with col1:
                st.markdown(content)
            with col2:
                try:
                    st.image(image_url, use_container_width=True)
                except Exception:
                    st.warning("Could not load image")
                    st.markdown(f"*Image URL: {image_url}*")
        else:
            st.markdown(content)
    else:
        # Display in expander
        with st.expander(f"📚 {title}", expanded=False):
            if image_url:
                col1, col2 = st.columns(columns)
                with col1:
                    st.markdown(content)
                with col2:
                    try:
                        st.image(image_url, use_container_width=True)
                    except Exception:
                        st.warning("Could not load image")
                        st.markdown(f"*Image URL: {image_url}*")
            else:
                st.markdown(content)


def key_points_section(
    points: list[str],
    title: str = "Key Points",
    icon: str = "🔑",
) -> None:
    """
    Display key points in a visually distinct section.

    Args:
        points: List of key point strings
        title: Section title
        icon: Icon to display before title
    """
    st.subheader(f"{icon} {title}")

    for i, point in enumerate(points, 1):
        with st.container(border=True):
            st.markdown(f"**{i}. {point}**")


def assumptions_section(
    assumptions: list[tuple[str, str]],
    title: str = "Assumptions",
    icon: str = "📋",
) -> None:
    """
    Display assumptions with descriptions.

    Args:
        assumptions: List of (assumption, description) tuples
        title: Section title
        icon: Icon to display before title
    """
    st.subheader(f"{icon} {title}")

    for assumption, description in assumptions:
        with st.container(border=True):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown(f"**{assumption}**")
            with col2:
                st.markdown(description)


# Example usage:
"""
# In a concept page:

from app.components import theory_section, key_points_section, assumptions_section

# Basic theory section
theory_section(
    title="Linear Regression",
    content=\"\"\"
    Linear regression models the relationship between a dependent variable
    and one or more independent variables using a linear approach...
    \"\"\",
    image_url="https://via.placeholder.com/400x250",
)

# Key points
key_points_section([
    "Models linear relationships between variables",
    "Minimizes sum of squared residuals",
    "Assumes homoscedasticity and independence",
])

# Assumptions
assumptions_section([
    ("Linearity", "Relationship between variables is linear"),
    ("Independence", "Observations are independent of each other"),
    ("Homoscedasticity", "Constant variance of errors"),
])
"""
