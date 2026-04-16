"""
References component for data science concept pages.
Displays academic references, books, and additional resources.
"""

import streamlit as st
from typing import List, Dict, Optional
from datetime import datetime


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
            ref_type = ref.get('type', 'other').lower()
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


def _display_reference_list(references: List[Dict[str, str]], show_abstracts: bool) -> None:
    """Helper function to display a list of references."""
    for i, ref in enumerate(references, 1):
        with st.container():
            # Citation style formatting
            authors = ref.get('authors', 'Unknown')
            year = ref.get('year', 'n.d.')
            title = ref.get('title', 'Untitled')
            journal = ref.get('journal', '')
            publisher = ref.get('publisher', '')
            url = ref.get('url', '')
            doi = ref.get('doi', '')
            
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
            if show_abstracts and 'abstract' in ref:
                with st.expander("Abstract", expanded=False):
                    st.markdown(ref['abstract'])
            
            # Tags (if available)
            if 'tags' in ref:
                tags = ref['tags'] if isinstance(ref['tags'], list) else [ref['tags']]
                tag_text = " | ".join([f"`{tag}`" for tag in tags])
                st.caption(f"Tags: {tag_text}")


def get_regression_references() -> List[Dict[str, str]]:
    """Get references for regression analysis."""
    return [
        {
            "type": "book",
            "authors": "James, G., Witten, D., Hastie, T., & Tibshirani, R.",
            "year": "2013",
            "title": "An Introduction to Statistical Learning: with Applications in R",
            "publisher": "Springer",
            "url": "https://www.statlearning.com/",
            "doi": "10.1007/978-1-4614-7138-7",
            "abstract": "This book provides an introduction to statistical learning methods. It is aimed for upper level undergraduate students, masters students and PhD students in the non-mathematical sciences. The book also contains a number of R labs with detailed explanations on how to implement the various methods in real life settings.",
            "tags": ["introductory", "R", "statistical learning"]
        },
        {
            "type": "book",
            "authors": "Hastie, T., Tibshirani, R., & Friedman, J.",
            "year": "2009",
            "title": "The Elements of Statistical Learning: Data Mining, Inference, and Prediction",
            "publisher": "Springer",
            "url": "https://hastie.su.domains/ElemStatLearn/",
            "doi": "10.1007/978-0-387-84858-7",
            "abstract": "This book describes the important ideas in these areas in a common conceptual framework. While the approach is statistical, the emphasis is on concepts rather than mathematics. Many examples are given, with a liberal use of color graphics.",
            "tags": ["advanced", "machine learning", "data mining"]
        },
        {
            "type": "paper",
            "authors": "Tibshirani, R.",
            "year": "1996",
            "title": "Regression Shrinkage and Selection via the Lasso",
            "journal": "Journal of the Royal Statistical Society: Series B (Methodological)",
            "url": "https://www.jstor.org/stable/2346178",
            "doi": "10.1111/j.2517-6161.1996.tb02080.x",
            "abstract": "We propose a new method for estimation in linear models. The 'lasso' minimizes the residual sum of squares subject to the sum of the absolute value of the coefficients being less than a constant.",
            "tags": ["lasso", "regularization", "feature selection"]
        },
        {
            "type": "article",
            "authors": "Hoerl, A. E., & Kennard, R. W.",
            "year": "1970",
            "title": "Ridge Regression: Biased Estimation for Nonorthogonal Problems",
            "journal": "Technometrics",
            "url": "https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634",
            "doi": "10.1080/00401706.1970.10488634",
            "abstract": "Ridge regression is a way to create a parsimonious model when the number of predictor variables in a set exceeds the number of observations, or when a data set exhibits multicollinearity.",
            "tags": ["ridge regression", "multicollinearity", "regularization"]
        },
        {
            "type": "online",
            "authors": "Scikit-learn Developers",
            "year": "2023",
            "title": "Linear Models - scikit-learn documentation",
            "publisher": "scikit-learn",
            "url": "https://scikit-learn.org/stable/modules/linear_model.html",
            "abstract": "Documentation for linear models in scikit-learn, including ordinary least squares, ridge regression, lasso, and elastic net.",
            "tags": ["python", "scikit-learn", "documentation"]
        }
    ]


def get_clustering_references() -> List[Dict[str, str]]:
    """Get references for clustering analysis."""
    return [
        {
            "type": "paper",
            "authors": "MacQueen, J.",
            "year": "1967",
            "title": "Some methods for classification and analysis of multivariate observations",
            "journal": "Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability",
            "url": "https://projecteuclid.org/proceedings/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Fifth-Berkeley-Symposium-on-Mathematical-Statistics-and/Chapter/Some-methods-for-classification-and-analysis-of-multivariate-observations/bsmsp/1200512992",
            "abstract": "The k-means algorithm is one of the most popular clustering algorithms. This paper introduces the original k-means algorithm.",
            "tags": ["k-means", "foundational", "clustering"]
        },
        {
            "type": "paper",
            "authors": "Ester, M., Kriegel, H.-P., Sander, J., & Xu, X.",
            "year": "1996",
            "title": "A density-based algorithm for discovering clusters in large spatial databases with noise",
            "journal": "Proceedings of the Second International Conference on Knowledge Discovery and Data Mining",
            "url": "https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf",
            "abstract": "DBSCAN, a density based clustering algorithm, is presented. It can discover clusters of arbitrary shape and is robust to noise.",
            "tags": ["DBSCAN", "density-based", "noise-resistant"]
        },
        {
            "type": "book",
            "authors": "Kaufman, L., & Rousseeuw, P. J.",
            "year": "1990",
            "title": "Finding Groups in Data: An Introduction to Cluster Analysis",
            "publisher": "Wiley",
            "doi": "10.1002/9780470316801",
            "abstract": "This book presents a comprehensive introduction to cluster analysis. It covers various clustering methods and provides practical guidance on their application.",
            "tags": ["cluster analysis", "comprehensive", "methods"]
        },
        {
            "type": "paper",
            "authors": "Rousseeuw, P. J.",
            "year": "1987",
            "title": "Silhouettes: a graphical aid to the interpretation and validation of cluster analysis",
            "journal": "Journal of Computational and Applied Mathematics",
            "url": "https://www.sciencedirect.com/science/article/pii/0377042787901257",
            "doi": "10.1016/0377-0427(87)90125-7",
            "abstract": "A graphical display is proposed for partitioning techniques, where each cluster is represented by a so-called silhouette, which is based on the comparison of its tightness and separation.",
            "tags": ["silhouette", "validation", "visualization"]
        },
        {
            "type": "online",
            "authors": "Scikit-learn Developers",
            "year": "2023",
            "title": "Clustering - scikit-learn documentation",
            "publisher": "scikit-learn",
            "url": "https://scikit-learn.org/stable/modules/clustering.html",
            "abstract": "Documentation for clustering algorithms in scikit-learn, including k-means, DBSCAN, hierarchical clustering, and more.",
            "tags": ["python", "scikit-learn", "documentation"]
        }
    ]


def get_general_data_science_references() -> List[Dict[str, str]]:
    """Get general data science references."""
    return [
        {
            "type": "book",
            "authors": "Wickham, H., & Grolemund, G.",
            "year": "2016",
            "title": "R for Data Science: Import, Tidy, Transform, Visualize, and Model Data",
            "publisher": "O'Reilly Media",
            "url": "https://r4ds.had.co.nz/",
            "abstract": "This book will teach you how to do data science with R: You'll learn how to get your data into R, get it into the most useful structure, transform it, visualise it and model it.",
            "tags": ["R", "tidyverse", "data visualization"]
        },
        {
            "type": "book",
            "authors": "McKinney, W.",
            "year": "2017",
            "title": "Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython",
            "publisher": "O'Reilly Media",
            "url": "https://wesmckinney.com/book/",
            "abstract": "This book is concerned with the nuts and bolts of manipulating, processing, cleaning, and crunching data in Python. It is also a practical, modern introduction to scientific computing in Python.",
            "tags": ["python", "pandas", "data wrangling"]
        },
        {
            "type": "online",
            "authors": "Various Contributors",
            "year": "2023",
            "title": "Towards Data Science",
            "publisher": "Medium",
            "url": "https://towardsdatascience.com/",
            "abstract": "A Medium publication sharing concepts, ideas, and codes in data science, machine learning, and AI.",
            "tags": ["blog", "tutorials", "community"]
        },
        {
            "type": "online",
            "authors": "Kaggle Team",
            "year": "2023",
            "title": "Kaggle Learn",
            "publisher": "Kaggle",
            "url": "https://www.kaggle.com/learn",
            "abstract": "Free micro-courses on data science and machine learning topics, with hands-on exercises and competitions.",
            "tags": ["courses", "hands-on", "competitions"]
        }
    ]


def search_references(query: str, references: List[Dict[str, str]]) -> List[Dict[str, str]]:
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
            ref.get('authors', '').lower(),
            ref.get('title', '').lower(),
            ref.get('abstract', '').lower(),
            ref.get('journal', '').lower(),
            ref.get('publisher', '').lower(),
            ' '.join(ref.get('tags', []))
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
    
    all_refs = get_regression_references() + get_clustering_references() + get_general_data_science_references()
    
    if search_query:
        filtered_refs = search_references(search_query, all_refs)
        st.write(f"Found {len(filtered_refs)} references matching '{search_query}'")
        display_references(filtered_refs, group_by_type=True, show_abstracts=True)
    else:
        st.subheader("Regression References")
        display_references(get_regression_references(), group_by_type=True, show_abstracts=False)
        
        st.subheader("Clustering References")
        display_references(get_clustering_references(), group_by_type=False, show_abstracts=True)
        
        st.subheader("General Data Science References")
        display_references(get_general_data_science_references(), group_by_type=True, show_abstracts=False)