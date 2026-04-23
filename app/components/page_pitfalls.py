"""
Pitfalls component for data science concept pages.
Displays common mistakes and how to avoid them.
"""

import streamlit as st
from typing import List, Dict, Optional


def display_pitfalls(
    pitfalls: List[Dict[str, str]],
    title: str = "Common Pitfalls & How to Avoid Them",
    show_severity: bool = True,
    show_solutions: bool = True,
) -> None:
    """
    Display a list of common pitfalls with severity indicators and solutions.

    Args:
        pitfalls: List of dictionaries with 'title', 'description', 'severity', and 'solution' keys
        title: Section title
        show_severity: Whether to show severity indicators (High/Medium/Low)
        show_solutions: Whether to show solution sections
    """
    if not pitfalls:
        st.info("No pitfalls documented for this concept.")
        return

    st.header(title)
    st.markdown("---")

    for i, pitfall in enumerate(pitfalls, 1):
        with st.container():
            col1, col2 = st.columns([3, 1])

            with col1:
                st.subheader(f"{i}. {pitfall['title']}")
                st.markdown(pitfall["description"])

            with col2:
                if show_severity and "severity" in pitfall:
                    severity = pitfall["severity"].lower()
                    if severity == "high":
                        st.error("**High Severity**")
                    elif severity == "medium":
                        st.warning("**Medium Severity**")
                    elif severity == "low":
                        st.info("**Low Severity**")
                    else:
                        st.write(f"**{pitfall['severity']}**")

            if show_solutions and "solution" in pitfall:
                with st.expander("**How to Avoid This Pitfall**", expanded=False):
                    if isinstance(pitfall["solution"], list):
                        for solution_item in pitfall["solution"]:
                            st.markdown(f"- {solution_item}")
                    else:
                        st.markdown(pitfall["solution"])

                    # Add optional tips if present
                    if "tips" in pitfall:
                        st.markdown("**Additional Tips:**")
                        if isinstance(pitfall["tips"], list):
                            for tip in pitfall["tips"]:
                                st.markdown(f"- {tip}")
                        else:
                            st.markdown(pitfall["tips"])

            st.markdown("---")


# Example usage in a Streamlit app:
if __name__ == "__main__":
    st.title("Pitfalls Component Demo")
    
    # Import data functions from data modules
    from app.data.regression import get_regression_pitfalls
    from app.data.clustering import get_clustering_pitfalls
    from app.data.general import get_classification_pitfalls
    
    st.subheader("Regression Pitfalls")
    display_pitfalls(get_regression_pitfalls())

    st.subheader("Clustering Pitfalls")
    display_pitfalls(get_clustering_pitfalls(), show_severity=True)

    st.subheader("Classification Pitfalls (Solutions Hidden)")
    display_pitfalls(get_classification_pitfalls(), show_solutions=False)
