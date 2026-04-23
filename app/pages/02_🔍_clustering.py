"""
Clustering concept page.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

from app.components import (
    theory_section,
    math_equation,
    code_tabs,
    display_references,
    clustering_demo,
)
from app.data.clustering import get_clustering_references
from app.logic.clustering import (
    generate_clustering_data,
    generate_complex_clustering_data,
    apply_kmeans,
    apply_dbscan,
    apply_hierarchical,
    apply_gmm,
    find_optimal_k,
    visualize_clusters_2d,
    visualize_clusters_3d,
    calculate_clustering_metrics,
)
from app.state import get_state


def main() -> None:
    """Main page function."""
    # Page header
    st.title("🔍 Clustering Analysis")
    st.caption(
        "Unsupervised learning for discovering patterns and grouping similar data points"
    )

    # Track page visit
    state = get_state()
    state.add_to_history("clustering")
    state.current_model = "clustering"

    # 1. Concept Overview
    theory_section(
        title="Concept Overview",
        content="""
        **Clustering** is an unsupervised learning technique that groups similar data points 
        together based on their characteristics, without prior knowledge of group labels.
        
        The goal is to partition data into clusters where points within the same cluster 
        are more similar to each other than to points in other clusters. This helps 
        discover inherent structures, patterns, and relationships in the data.
        
        ### Key Characteristics:
        - **Unsupervised learning** - No labeled training data required
        - **Exploratory analysis** - Discovers hidden patterns and structures
        - **Diverse algorithms** - Different approaches for different data types
        - **Visualization friendly** - Results are often easy to visualize
        
        Clustering is widely used for customer segmentation, anomaly detection, 
        image segmentation, document organization, and many other applications.
        """,
        image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/0/0f/Cluster-2.svg/800px-Cluster-2.svg.png",
        columns=(2, 1),
    )

    # Mathematical formulation (not in expander since content is short)
    with st.expander("🧮 Mathematical Formulation", expanded=False):
        math_equation(
            equation=r"J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2",
            variables={
                "J": "Objective function (within-cluster sum of squares)",
                "k": "Number of clusters",
                "C_i": "Set of points in cluster i",
                "x": "Data point",
                "\\mu_i": "Centroid of cluster i",
            },
            title="K-means Objective Function",
            icon="📐",
            expandable=False,
        )

        # Additional metrics (not in expander since content is short)
        st.markdown("**Silhouette Score:**")
        st.latex(r"s(i) = \frac{b(i) - a(i)}{\max\{a(i), b(i)\}}")
        st.markdown("Where:")
        st.markdown(
            r"- $a(i)$: Average distance from point $i$ to other points in same cluster"
        )
        st.markdown(
            r"- $b(i)$: Average distance from point $i$ to points in nearest neighboring cluster"
        )

        st.markdown("**DBSCAN Density:**")
        st.latex(r"N_\epsilon(p) = \{q \in D \mid \text{dist}(p, q) \leq \epsilon\}")
        st.markdown("Where:")
        st.markdown(r"- $N_\epsilon(p)$: $\epsilon$-neighborhood of point $p$")
        st.markdown(r"- $\epsilon$: Maximum distance between points")
        st.markdown(r"- $\text{minPts}$: Minimum points to form a dense region")

    # 2. Interactive Demo
    clustering_demo()

    # 3. Implementation Examples
    with st.expander("💻 Implementation Examples", expanded=True):
        code_tabs(
            {
                "K-means with scikit-learn": """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Generate sample data
    X = np.random.randn(300, 2)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply K-means
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Get results
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    inertia = kmeans.inertia_

    print(f"Found {len(np.unique(labels))} clusters")
    print(f"Inertia: {inertia:.2f}")
    """,
                "DBSCAN for density-based clustering": """
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Generate sample data
    X = np.random.randn(300, 2)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)

    # Analyze results
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)

    print(f"Found {n_clusters} clusters")
    print(f"Noise points: {n_noise}")
    print(f"Labels: {unique_labels}")
    """,
                "Hierarchical clustering with different linkages": """
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Generate sample data
    X = np.random.randn(300, 2)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try different linkage methods
    for linkage in ['ward', 'complete', 'average', 'single']:
        hierarchical = AgglomerativeClustering(
            n_clusters=4, 
            linkage=linkage
        )
        labels = hierarchical.fit_predict(X_scaled)
        print(f"{linkage} linkage - Labels: {np.unique(labels)}")
    """,
                "Gaussian Mixture Models for probabilistic clustering": """
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Generate sample data
    X = np.random.randn(300, 2)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply GMM
    gmm = GaussianMixture(
        n_components=4,
        covariance_type='full',
        random_state=42
    )
    labels = gmm.fit_predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)

    print(f"Found {len(np.unique(labels))} clusters")
    print(f"BIC: {gmm.bic(X_scaled):.1f}")
    print(f"AIC: {gmm.aic(X_scaled):.1f}")
    print(f"Converged: {gmm.converged_}")
    """,
            }
        )

    # 4. Real-World Applications (in expander with tabs inside)
    with st.expander("💼 Real-World Applications", expanded=False):
        applications_data = {
            "Customer Segmentation": {
                "description": "Group customers based on purchasing behavior, demographics, and preferences.",
                "details": "Enables targeted marketing, personalized recommendations, and improved customer retention strategies.",
                "examples": [
                    "E-commerce customer segmentation for personalized offers",
                    "Banking customer profiling for tailored financial products",
                    "Subscription service user groups for content recommendations",
                ],
            },
            "Image Segmentation": {
                "description": "Group pixels in images based on color, texture, or other features.",
                "details": "Used in computer vision for object detection, medical imaging analysis, and autonomous vehicle systems.",
                "examples": [
                    "Medical image analysis for tumor detection",
                    "Satellite image classification for land use mapping",
                    "Facial recognition systems for security applications",
                ],
            },
            "Anomaly Detection": {
                "description": "Identify unusual patterns or outliers in data that deviate from normal behavior.",
                "details": "Critical for fraud detection, network security monitoring, and quality control in manufacturing.",
                "examples": [
                    "Credit card fraud detection systems",
                    "Network intrusion detection for cybersecurity",
                    "Manufacturing defect detection in production lines",
                ],
            },
            "Document Clustering": {
                "description": "Group similar documents for organization, retrieval, and topic modeling.",
                "details": "Used in information retrieval systems, recommendation engines, and content management platforms.",
                "examples": [
                    "News article categorization for media platforms",
                    "Research paper organization for academic databases",
                    "Customer support ticket grouping for efficient resolution",
                ],
            },
        }

        app_tabs = st.tabs(list(applications_data.keys()))

        for tab, (app_name, app_info) in zip(app_tabs, applications_data.items()):
            with tab:
                st.subheader(app_name)
                st.markdown(f"**Description:** {app_info['description']}")
                st.markdown(f"**Details:** {app_info['details']}")
                st.markdown("**Examples:**")
                for example in app_info["examples"]:
                    st.markdown(f"- {example}")

    # 5. Common Pitfalls & Fixes (in expander with tabs inside)
    with st.expander("⚠️ Common Pitfalls & Fixes", expanded=False):
        pitfalls_data = {
            "Choosing Wrong Algorithm": {
                "problem": "Selecting an inappropriate clustering algorithm for the data type or problem.",
                "detection": "Poor clustering results, unnatural cluster shapes, or inability to find meaningful patterns.",
                "solution": "Understand algorithm assumptions: K-means for spherical clusters, DBSCAN for density-based, hierarchical for hierarchical structures.",
            },
            "Incorrect Distance Metric": {
                "problem": "Using inappropriate distance measures that don't capture data similarity correctly.",
                "detection": "Clusters don't reflect natural groupings, similar points end up in different clusters.",
                "solution": "Choose metric based on data type: Euclidean for continuous, Manhattan for grid-like, cosine for text, Jaccard for binary.",
            },
            "Scale Sensitivity": {
                "problem": "Algorithms like K-means are sensitive to feature scales, giving undue importance to features with larger ranges.",
                "detection": "Features with larger ranges dominate clustering, distorting results.",
                "solution": "Always scale features (standardization or normalization) before applying distance-based algorithms.",
            },
            "Determining Optimal K": {
                "problem": "Difficulty in choosing the right number of clusters, especially for algorithms like K-means.",
                "detection": "Unclear elbow point in elbow method, ambiguous silhouette scores.",
                "solution": "Use multiple methods: elbow method, silhouette analysis, gap statistic, and domain knowledge.",
            },
            "Handling Noise & Outliers": {
                "problem": "Noise points can distort cluster boundaries and centroids, especially in density-based methods.",
                "detection": "Many points classified as noise, unstable cluster boundaries.",
                "solution": "Use robust algorithms like DBSCAN, pre-process data to remove outliers, or use noise-handling variants.",
            },
        }

        pitfall_tabs = st.tabs(list(pitfalls_data.keys()))

        for tab, (pitfall_name, pitfall_info) in zip(
            pitfall_tabs, pitfalls_data.items()
        ):
            with tab:
                st.subheader(pitfall_name)
                st.markdown(f"**Problem:** {pitfall_info['problem']}")
                st.markdown(f"**How to Detect:** {pitfall_info['detection']}")
                st.markdown(f"**Solution:** {pitfall_info['solution']}")

    # 6. References & Further Reading (in expander)
    with st.expander("📚 References & Further Reading", expanded=False):
        display_references(get_clustering_references())

    # Footer
    st.markdown("---")
    st.caption(
        "Clustering Analysis Concept • "
        "Experiment with different algorithms and datasets • "
        "Next: Explore other machine learning concepts in the platform"
    )


if __name__ == "__main__":
    main()
