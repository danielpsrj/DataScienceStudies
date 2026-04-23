"""
References component for data science concept pages.
Displays academic references, books, and additional resources.
"""

import streamlit as st
from typing import List, Dict, Optional


def display_references(
    references: List[Dict[str, str]],
    title: str = "References & Further Reading",
    group_by_type: bool = True,
    show_abstracts: bool = False,
) -> None:
    """
    Display a list of references with optional grouping and abstracts.

    Args:
        references: List of dictionaries with reference information
        title: Section title
        group_by_type: Whether to group references by type (book, paper, article, etc.)
        show_abstracts: Whether to show abstracts in expandable sections
    """
    if not references:
        st.info("No references available for this concept.")
        return

    st.header(title)
    st.markdown("---")

    if group_by_type:
        # Group references by type
        grouped_refs = {}
        for ref in references:
            ref_type = ref.get("type", "other").lower()
            if ref_type not in grouped_refs:
                grouped_refs[ref_type] = []
            grouped_refs[ref_type].append(ref)

        # Display by group
        for ref_type, type_refs in grouped_refs.items():
            st.subheader(f"{ref_type.title()}s")
            _display_reference_list(type_refs, show_abstracts)
            st.markdown("---")
    else:
        _display_reference_list(references, show_abstracts)


def _display_reference_list(
    references: List[Dict[str, str]], show_abstracts: bool
) -> None:
    """Helper function to display a list of references."""
    for i, ref in enumerate(references, 1):
        with st.container():
            # Citation style formatting
            authors = ref.get("authors", "Unknown")
            year = ref.get("year", "n.d.")
            title = ref.get("title", "Untitled")
            journal = ref.get("journal", "")
            publisher = ref.get("publisher", "")
            url = ref.get("url", "")
            doi = ref.get("doi", "")

            # Format citation
            citation = f"**{authors} ({year}).** *{title}*"
            if journal:
                citation += f". {journal}"
            if publisher and not journal:
                citation += f". {publisher}"

            st.markdown(f"{i}. {citation}")

            # DOI and URL links
            link_cols = st.columns([1, 1])
            with link_cols[0]:
                if doi:
                    st.markdown(f"DOI: [{doi}](https://doi.org/{doi})")
            with link_cols[1]:
                if url:
                    st.markdown(f"[📖 Read Online]({url})")

            # Abstract (if available and requested)
            if show_abstracts and "abstract" in ref:
                with st.expander("Abstract", expanded=False):
                    st.markdown(ref["abstract"])

            # Tags (if available)
            if "tags" in ref:
                tags = ref["tags"] if isinstance(ref["tags"], list) else [ref["tags"]]
                tag_text = " | ".join([f"`{tag}`" for tag in tags])
                st.caption(f"Tags: {tag_text}")


def search_references(
    query: str, references: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """
    Search references by query string.

    Args:
        query: Search query string
        references: List of references to search through

    Returns:
        Filtered list of references matching the query
    """
    if not query:
        return references

    query_lower = query.lower()
    results = []

    for ref in references:
        # Search in various fields
        search_fields = [
            ref.get("authors", "").lower(),
            ref.get("title", "").lower(),
            ref.get("abstract", "").lower(),
            ref.get("journal", "").lower(),
            ref.get("publisher", "").lower(),
            " ".join(ref.get("tags", [])),
        ]

        # Check if query appears in any field
        if any(query_lower in field for field in search_fields if field):
            results.append(ref)

    return results


# Example usage in a Streamlit app:
if __name__ == "__main__":
    st.title("References Component Demo")

    # Search functionality demo
    st.subheader("Search References")
    search_query = st.text_input("Search references (authors, title, tags):")

    # Import data functions from data modules
    from app.data.regression import get_regression_references
    from app.data.clustering import get_clustering_references
    from app.data.general import get_general_data_science_references
    
    all_refs = (
        get_regression_references()
        + get_clustering_references()
        + get_general_data_science_references()
    )

    if search_query:
        filtered_refs = search_references(search_query, all_refs)
        st.write(f"Found {len(filtered_refs)} references matching '{search_query}'")
        display_references(filtered_refs, group_by_type=True, show_abstracts=True)
    else:
        st.subheader("Regression References")
        display_references(
            get_regression_references(), group_by_type=True, show_abstracts=False
        )

        st.subheader("Clustering References")
        display_references(
            get_clustering_references(), group_by_type=False, show_abstracts=True
        )

        st.subheader("General Data Science References")
        display_references(
            get_general_data_science_references(),
            group_by_type=True,
            show_abstracts=False,
        )