"""
Applications component for data science concept pages.
Displays real-world applications of the concept being taught.
"""

import streamlit as st
from typing import List, Dict, Optional


def display_applications(
    applications: List[Dict[str, str]],
    title: str = "Real-World Applications",
    columns: int = 2,
    expand_all: bool = False,
) -> None:
    """
    Display a grid of real-world applications with descriptions.

    Args:
        applications: List of dictionaries with 'title' and 'description' keys
        title: Section title
        columns: Number of columns in the grid
        expand_all: Whether to expand all application cards by default
    """
    if not applications:
        st.info("No applications available for this concept.")
        return

    st.header(title)
    st.markdown("---")

    # Create columns for the grid
    cols = st.columns(columns)

    for i, app in enumerate(applications):
        col_idx = i % columns
        with cols[col_idx]:
            with st.expander(f"**{app['title']}**", expanded=expand_all):
                st.markdown(app["description"])

                # Add optional details if present
                if "details" in app:
                    st.markdown("**Details:**")
                    st.markdown(app["details"])

                # Add optional examples if present
                if "examples" in app:
                    st.markdown("**Examples:**")
                    if isinstance(app["examples"], list):
                        for example in app["examples"]:
                            st.markdown(f"- {example}")
                    else:
                        st.markdown(app["examples"])


# Example usage in a Streamlit app:
if __name__ == "__main__":
    st.title("Applications Component Demo")
    
    # Import data functions from data modules
    from app.data.regression import get_regression_applications
    from app.data.clustering import get_clustering_applications
    from app.data.general import get_time_series_applications
    
    st.subheader("Regression Applications")
    display_applications(get_regression_applications())

    st.subheader("Clustering Applications")
    display_applications(get_clustering_applications(), columns=3)

    st.subheader("Time Series Applications (Expanded)")
    display_applications(get_time_series_applications(), expand_all=True)
