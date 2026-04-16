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
    get_clustering_references,
)
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
        icon="🧮",
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
    st.header("🎮 Interactive Demo")

    # Demo controls at the top (horizontal layout)
    st.subheader("⚙️ Clustering Parameters")

    # Create horizontal columns for controls
    param_col1, param_col2, param_col3, param_col4 = st.columns(4)

    with param_col1:
        dataset_type = st.selectbox(
            "Dataset Type",
            ["blobs", "moons", "circles", "anisotropic", "random"],
            index=0,
            help="Type of synthetic dataset to generate",
        )

        n_samples = st.slider(
            "Sample Size",
            min_value=50,
            max_value=1000,
            value=300,
            step=50,
            help="Number of data points to generate",
        )

    with param_col2:
        algorithm = st.selectbox(
            "Clustering Algorithm",
            ["K-means", "DBSCAN", "Hierarchical", "GMM"],
            index=0,
            help="Algorithm to apply to the data",
        )

        scale_data = st.checkbox(
            "Scale Features",
            value=True,
            help="Standardize features before clustering",
        )

    with param_col3:
        # Dataset-specific parameters
        if dataset_type == "blobs":
            n_clusters = st.slider(
                "True Clusters",
                min_value=2,
                max_value=10,
                value=4,
                help="Actual number of clusters in the data",
            )
            cluster_std = st.slider(
                "Cluster Spread",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Standard deviation of clusters",
            )
        else:
            noise = st.slider(
                "Noise Level",
                min_value=0.01,
                max_value=0.3,
                value=0.05,
                step=0.01,
                help="Amount of noise in the dataset",
            )

    with param_col4:
        # Algorithm-specific parameters
        if algorithm == "K-means":
            k_value = st.slider(
                "Number of Clusters (k)",
                min_value=2,
                max_value=10,
                value=4,
                help="Number of clusters to find",
            )
            random_state = st.number_input(
                "Random Seed",
                min_value=0,
                max_value=100,
                value=42,
                help="Random seed for reproducibility",
            )
        elif algorithm == "DBSCAN":
            eps = st.slider(
                "Epsilon (ε)",
                min_value=0.1,
                max_value=2.0,
                value=0.5,
                step=0.1,
                help="Maximum distance between points",
            )
            min_samples = st.slider(
                "Min Samples",
                min_value=2,
                max_value=20,
                value=5,
                help="Minimum points to form dense region",
            )
        elif algorithm == "Hierarchical":
            linkage_method = st.selectbox(
                "Linkage Method",
                ["ward", "complete", "average", "single"],
                index=0,
                help="Linkage criterion",
            )
            n_clusters_h = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=10,
                value=4,
                help="Number of clusters to find",
            )
        elif algorithm == "GMM":
            n_components = st.slider(
                "Number of Components",
                min_value=2,
                max_value=10,
                value=4,
                help="Number of Gaussian mixture components",
            )
            covariance_type = st.selectbox(
                "Covariance Type",
                ["full", "tied", "diag", "spherical"],
                index=0,
                help="Type of covariance parameters",
            )

    # Run button centered below controls
    run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
    with run_col2:
        if st.button(
            "🚀 Run Clustering Analysis", type="primary", use_container_width=True
        ):
            with st.spinner("Generating data and applying clustering algorithm..."):
                # Generate data based on selected parameters
                if dataset_type == "blobs":
                    X, y_true = generate_clustering_data(
                        n_samples=n_samples,
                        n_clusters=n_clusters,
                        cluster_std=cluster_std,
                        random_state=42,
                    )
                elif dataset_type in ["moons", "circles", "anisotropic"]:
                    X, y_true = generate_complex_clustering_data(
                        n_samples=n_samples,
                        dataset_type=dataset_type,
                        noise=noise if dataset_type != "anisotropic" else 0.05,
                        random_state=42,
                    )
                else:  # random
                    X = np.random.randn(n_samples, 2)
                    y_true = np.zeros(n_samples)

                # Apply selected algorithm
                if algorithm == "K-means":
                    result = apply_kmeans(
                        X,
                        n_clusters=k_value,
                        random_state=random_state,
                        scale_data=scale_data,
                    )
                    labels = result["labels"]
                    centers = result["centers"]

                elif algorithm == "DBSCAN":
                    result = apply_dbscan(
                        X, eps=eps, min_samples=min_samples, scale_data=scale_data
                    )
                    labels = result["labels"]
                    centers = None

                elif algorithm == "Hierarchical":
                    result = apply_hierarchical(
                        X,
                        n_clusters=n_clusters_h,
                        linkage=linkage_method,
                        scale_data=scale_data,
                    )
                    labels = result["labels"]
                    centers = None

                elif algorithm == "GMM":
                    result = apply_gmm(
                        X,
                        n_components=n_components,
                        covariance_type=covariance_type,
                        random_state=42,
                        scale_data=scale_data,
                    )
                    labels = result["labels"]
                    centers = None

                # Create visualizations
                tab1, tab2, tab3 = st.tabs(["2D Clusters", "3D View", "True Labels"])

                with tab1:
                    fig = visualize_clusters_2d(
                        X,
                        labels,
                        title=f"{algorithm} Clustering Results",
                        centers=centers,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Show cluster statistics
                    unique_labels = np.unique(labels)
                    n_clusters_found = len(unique_labels) - (
                        1 if -1 in unique_labels else 0
                    )
                    st.metric("Clusters Found", n_clusters_found)

                    if algorithm == "DBSCAN" and -1 in unique_labels:
                        n_noise = np.sum(labels == -1)
                        st.metric("Noise Points", n_noise)

                with tab2:
                    if X.shape[1] >= 2:
                        # Add a third dimension if needed
                        if X.shape[1] == 2:
                            X_3d = np.hstack([X, np.zeros((X.shape[0], 1))])
                        else:
                            X_3d = X[:, :3]

                        fig_3d = visualize_clusters_3d(
                            X_3d, labels, title=f"{algorithm} - 3D View"
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                    else:
                        st.info("3D visualization requires at least 2 dimensions")

                with tab3:
                    if dataset_type != "random":
                        fig_true = visualize_clusters_2d(
                            X, y_true, title="True Cluster Labels", centers=None
                        )
                        st.plotly_chart(fig_true, use_container_width=True)
                        st.metric("True Clusters", len(np.unique(y_true)))
                    else:
                        st.info("Random data has no true labels")

                # Performance metrics
                st.subheader("📊 Performance Metrics")
                metrics = calculate_clustering_metrics(X, labels)

                if metrics:
                    col_metric1, col_metric2 = st.columns(2)

                    with col_metric1:
                        if "silhouette_score" in metrics and not np.isnan(
                            metrics["silhouette_score"]
                        ):
                            st.metric(
                                "Silhouette Score",
                                f"{metrics['silhouette_score']:.3f}",
                                help="Higher is better (-1 to 1)",
                            )

                        if "calinski_harabasz_score" in metrics and not np.isnan(
                            metrics["calinski_harabasz_score"]
                        ):
                            st.metric(
                                "Calinski-Harabasz",
                                f"{metrics['calinski_harabasz_score']:.1f}",
                                help="Higher is better",
                            )

                    with col_metric2:
                        if "davies_bouldin_score" in metrics and not np.isnan(
                            metrics["davies_bouldin_score"]
                        ):
                            st.metric(
                                "Davies-Bouldin",
                                f"{metrics['davies_bouldin_score']:.3f}",
                                help="Lower is better (0 to ∞)",
                            )

                        if algorithm == "K-means" and "inertia" in result:
                            st.metric(
                                "Inertia",
                                f"{result['inertia']:.1f}",
                                help="Within-cluster sum of squares (lower is better)",
                            )
                else:
                    st.info("Metrics require at least 2 clusters")

                # Algorithm-specific metrics
                if algorithm == "DBSCAN":
                    st.subheader("DBSCAN Stats")
                    st.write(f"**Clusters:** {result['n_clusters']}")
                    st.write(f"**Noise Points:** {result['n_noise']}")
                    st.write(f"**Epsilon:** {result['eps']}")
                    st.write(f"**Min Samples:** {result['min_samples']}")

                elif algorithm == "GMM":
                    st.subheader("GMM Stats")
                    st.write(f"**Components:** {result['n_components']}")
                    st.write(f"**Covariance:** {result['covariance_type']}")
                    st.write(f"**BIC:** {result['bic']:.1f}")
                    st.write(f"**AIC:** {result['aic']:.1f}")
                    st.write(f"**Converged:** {result['converged']}")

                # Optimal K analysis (for K-means)
                if algorithm == "K-means":
                    st.subheader("🔍 Optimal K Analysis")

                    if st.button("Find Optimal K", type="secondary"):
                        with st.spinner("Analyzing optimal number of clusters..."):
                            optimal_k_result = find_optimal_k(
                                X, k_range=(2, 10), scale_data=scale_data
                            )

                            # Store in session state
                            st.session_state.optimal_k_result = optimal_k_result

                    if "optimal_k_result" in st.session_state:
                        result_opt = st.session_state.optimal_k_result
                        st.success(
                            f"Optimal k (Silhouette): {result_opt['optimal_k_silhouette']}"
                        )
                        st.info(f"Optimal k (Elbow): {result_opt['optimal_k_elbow']}")

                        # Plot elbow curve
                        k_values = list(result_opt["results"].keys())
                        inertias = [
                            result_opt["results"][k]["inertia"] for k in k_values
                        ]
                        silhouette_scores = [
                            result_opt["results"][k]["silhouette_score"]
                            for k in k_values
                        ]

                        fig_elbow = go.Figure()
                        fig_elbow.add_trace(
                            go.Scatter(
                                x=k_values,
                                y=inertias,
                                mode="lines+markers",
                                name="Inertia",
                                line=dict(color="blue", width=2),
                            )
                        )

                        # Add silhouette scores on secondary y-axis
                        fig_elbow.add_trace(
                            go.Scatter(
                                x=k_values,
                                y=silhouette_scores,
                                mode="lines+markers",
                                name="Silhouette Score",
                                yaxis="y2",
                                line=dict(color="green", width=2, dash="dash"),
                            )
                        )

                        fig_elbow.update_layout(
                            title="Elbow Method & Silhouette Analysis",
                            xaxis_title="Number of Clusters (k)",
                            yaxis_title="Inertia",
                            yaxis2=dict(
                                title="Silhouette Score", overlaying="y", side="right"
                            ),
                            hovermode="x unified",
                            showlegend=True,
                        )

                        st.plotly_chart(fig_elbow, use_container_width=True)

    # 3. Implementation Examples
    st.header("💻 Implementation Examples")

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
    st.header("📚 References & Further Reading")

    with st.expander("Click to view references", expanded=False):
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
