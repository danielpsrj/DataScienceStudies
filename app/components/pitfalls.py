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
                st.markdown(pitfall['description'])
            
            with col2:
                if show_severity and 'severity' in pitfall:
                    severity = pitfall['severity'].lower()
                    if severity == 'high':
                        st.error("**High Severity**")
                    elif severity == 'medium':
                        st.warning("**Medium Severity**")
                    elif severity == 'low':
                        st.info("**Low Severity**")
                    else:
                        st.write(f"**{pitfall['severity']}**")
            
            if show_solutions and 'solution' in pitfall:
                with st.expander("**How to Avoid This Pitfall**", expanded=False):
                    if isinstance(pitfall['solution'], list):
                        for solution_item in pitfall['solution']:
                            st.markdown(f"- {solution_item}")
                    else:
                        st.markdown(pitfall['solution'])
                    
                    # Add optional tips if present
                    if 'tips' in pitfall:
                        st.markdown("**Additional Tips:**")
                        if isinstance(pitfall['tips'], list):
                            for tip in pitfall['tips']:
                                st.markdown(f"- {tip}")
                        else:
                            st.markdown(pitfall['tips'])
            
            st.markdown("---")


def get_regression_pitfalls() -> List[Dict[str, str]]:
    """Get common pitfalls for regression analysis."""
    return [
        {
            "title": "Overfitting",
            "description": "Creating a model that fits the training data too closely, capturing noise rather than underlying patterns.",
            "severity": "High",
            "solution": [
                "Use cross-validation to evaluate model performance",
                "Apply regularization techniques (L1/L2 regularization)",
                "Simplify the model by reducing features",
                "Use more training data when possible"
            ],
            "tips": "Always validate your model on unseen data. A model that performs perfectly on training data but poorly on test data is likely overfit."
        },
        {
            "title": "Multicollinearity",
            "description": "When predictor variables are highly correlated with each other, making it difficult to determine individual effects.",
            "severity": "Medium",
            "solution": [
                "Calculate Variance Inflation Factor (VIF) to detect multicollinearity",
                "Remove highly correlated features",
                "Use dimensionality reduction techniques (PCA)",
                "Apply regularization methods"
            ],
            "tips": "Check correlation matrices before building models. Features with correlation > 0.8 or < -0.8 should be investigated."
        },
        {
            "title": "Ignoring Non-Linear Relationships",
            "description": "Assuming linear relationships when the true relationship is non-linear.",
            "severity": "Medium",
            "solution": [
                "Plot residuals to check for patterns",
                "Use polynomial features or interaction terms",
                "Consider non-linear models (decision trees, neural networks)",
                "Apply transformations (log, square root) to variables"
            ],
            "tips": "Always visualize the relationship between variables before assuming linearity. Scatter plots with trend lines can reveal non-linear patterns."
        },
        {
            "title": "Outliers Unduly Influencing Results",
            "description": "Extreme values disproportionately affecting regression coefficients and predictions.",
            "severity": "High",
            "solution": [
                "Identify outliers using box plots or statistical methods (IQR)",
                "Consider robust regression methods",
                "Transform variables to reduce outlier impact",
                "Remove or winsorize outliers when justified"
            ],
            "tips": "Be cautious when removing outliers - understand why they exist before deciding to exclude them."
        },
        {
            "title": "Ignoring Assumptions",
            "description": "Failing to check regression assumptions (linearity, independence, homoscedasticity, normality).",
            "severity": "Medium",
            "solution": [
                "Check residual plots for patterns",
                "Test for heteroscedasticity (Breusch-Pagan test)",
                "Check for autocorrelation (Durbin-Watson test)",
                "Test normality of residuals (Q-Q plots, Shapiro-Wilk test)"
            ],
            "tips": "Always validate assumptions before interpreting results. Violated assumptions can lead to incorrect conclusions."
        }
    ]


def get_clustering_pitfalls() -> List[Dict[str, str]]:
    """Get common pitfalls for clustering analysis."""
    return [
        {
            "title": "Choosing Wrong Number of Clusters",
            "description": "Selecting too many or too few clusters, leading to over-segmentation or under-segmentation.",
            "severity": "High",
            "solution": [
                "Use elbow method to find optimal k",
                "Apply silhouette analysis",
                "Use gap statistic method",
                "Consider domain knowledge and business requirements"
            ],
            "tips": "There's no one-size-fits-all answer. Different methods may suggest different optimal k values."
        },
        {
            "title": "Ignoring Feature Scaling",
            "description": "Using unscaled features when distance-based algorithms (like K-means) are sensitive to scale.",
            "severity": "High",
            "solution": [
                "Always scale features before clustering",
                "Use standardization (z-score normalization)",
                "Consider min-max scaling for bounded ranges",
                "Use algorithms less sensitive to scale (DBSCAN with appropriate parameters)"
            ],
            "tips": "Distance-based algorithms treat all dimensions equally. A feature with larger range will dominate the distance calculation."
        },
        {
            "title": "Assuming Clusters are Spherical",
            "description": "Using algorithms that assume spherical clusters (like K-means) for non-spherical data.",
            "severity": "Medium",
            "solution": [
                "Visualize data in 2D/3D to understand cluster shapes",
                "Use density-based clustering (DBSCAN) for arbitrary shapes",
                "Consider hierarchical clustering",
                "Use spectral clustering for complex structures"
            ],
            "tips": "K-means works well for spherical, equally sized clusters. For other shapes, explore different algorithms."
        },
        {
            "title": "Interpreting Clusters Without Validation",
            "description": "Assuming clusters have meaningful interpretation without proper validation.",
            "severity": "Medium",
            "solution": [
                "Validate clusters with domain experts",
                "Use internal validation metrics (silhouette score, Davies-Bouldin index)",
                "Compare with ground truth if available",
                "Test stability with different initializations"
            ],
            "tips": "Clusters are mathematical constructs, not necessarily meaningful business segments. Always validate with domain knowledge."
        },
        {
            "title": "Ignoring High-Dimensionality Issues",
            "description": "Applying clustering directly to high-dimensional data without dimensionality reduction.",
            "severity": "Medium",
            "solution": [
                "Apply PCA or t-SNE for dimensionality reduction",
                "Use feature selection to reduce dimensions",
                "Consider subspace clustering methods",
                "Use algorithms designed for high dimensions"
            ],
            "tips": "In high dimensions, distance measures become less meaningful (curse of dimensionality). Dimensionality reduction is often essential."
        }
    ]


def get_classification_pitfalls() -> List[Dict[str, str]]:
    """Get common pitfalls for classification analysis."""
    return [
        {
            "title": "Class Imbalance",
            "description": "When one class has significantly more samples than others, leading to biased models.",
            "severity": "High",
            "solution": [
                "Use resampling techniques (oversampling minority, undersampling majority)",
                "Apply class weights in algorithms",
                "Use appropriate evaluation metrics (precision, recall, F1-score)",
                "Try anomaly detection approaches for extreme imbalance"
            ],
            "tips": "Accuracy is misleading with imbalanced data. Always use metrics like precision, recall, and F1-score."
        },
        {
            "title": "Data Leakage",
            "description": "Allowing information from test/validation data to influence training, leading to overly optimistic results.",
            "severity": "High",
            "solution": [
                "Always split data before any preprocessing",
                "Use pipeline objects to prevent leakage",
                "Be careful with time-series data (use time-based splits)",
                "Validate preprocessing steps are fit only on training data"
            ],
            "tips": "If your model performs suspiciously well, check for data leakage. This is a common cause of unrealistic performance."
        },
        {
            "title": "Ignoring Probability Calibration",
            "description": "Treating model probabilities as true probabilities without calibration.",
            "severity": "Medium",
            "solution": [
                "Use calibration curves to assess probability quality",
                "Apply Platt scaling or isotonic regression",
                "Choose algorithms with naturally calibrated probabilities (like logistic regression)",
                "Use ensemble methods with probability calibration"
            ],
            "tips": "Some algorithms (like SVM, decision trees) produce poorly calibrated probabilities. Always check calibration if using probabilities for decision making."
        }
    ]


# Example usage in a Streamlit app:
if __name__ == "__main__":
    st.title("Pitfalls Component Demo")
    
    st.subheader("Regression Pitfalls")
    display_pitfalls(get_regression_pitfalls())
    
    st.subheader("Clustering Pitfalls")
    display_pitfalls(get_clustering_pitfalls(), show_severity=True)
    
    st.subheader("Classification Pitfalls (Solutions Hidden)")
    display_pitfalls(get_classification_pitfalls(), show_solutions=False)