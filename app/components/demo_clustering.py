"""
Clustering-specific demo component.
Contains demo logic specific to the Clustering page.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
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


def clustering_demo() -> None:
    """Interactive demo for clustering analysis."""
    with st.expander("🎮 Interactive Demo", expanded=True): 
        # Demo controls at the top (horizontal layout)
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
        if st.button("🚀 Run Clustering Analysis", type="primary", use_container_width=True):
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
                
                # Display results
                _display_clustering_results(X, labels, y_true, algorithm, result, centers)


def _display_clustering_results(X, labels, y_true, algorithm, result, centers):
    """Display clustering results."""
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
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
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
        if hasattr(y_true, 'shape'):  # Check if y_true is an array
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
            if "silhouette_score" in metrics and not np.isnan(metrics["silhouette_score"]):
                st.metric(
                    "Silhouette Score",
                    f"{metrics['silhouette_score']:.3f}",
                    help="Higher is better (-1 to 1)",
                )
            
            if "calinski_harabasz_score" in metrics and not np.isnan(metrics["calinski_harabasz_score"]):
                st.metric(
                    "Calinski-Harabasz",
                    f"{metrics['calinski_harabasz_score']:.1f}",
                    help="Higher is better",
                )
        
        with col_metric2:
            if "davies_bouldin_score" in metrics and not np.isnan(metrics["davies_bouldin_score"]):
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


def get_clustering_demo_code() -> str:
    """Get code snippet for the clustering demo."""
    return '''
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons, make_circles
import numpy as np

def run_clustering_demo(dataset_type='blobs', algorithm='K-means', n_samples=300, 
                        n_clusters=4, scale_data=True, **kwargs):
    # Generate data
    if dataset_type == 'blobs':
        X, y_true = make_blobs(n_samples=n_samples, n_features=2, 
                               centers=n_clusters, random_state=42)
    elif dataset_type == 'moons':
        X, y_true = make_moons(n_samples=n_samples, noise=0.05, random_state=42)
    elif dataset_type == 'circles':
        X, y_true = make_circles(n_samples=n_samples, noise=0.05, 
                                 factor=0.5, random_state=42)
    else:
        X = np.random.randn(n_samples, 2)
        y_true = None
    
    # Scale data if requested
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Apply clustering algorithm
    if algorithm == 'K-means':
        kmeans = KMeans(n_clusters=kwargs.get('k_value', 4), 
                       random_state=kwargs.get('random_state', 42),
                       n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        centers = scaler.inverse_transform(kmeans.cluster_centers_) if scale_data else kmeans.cluster_centers_
        result = {
            'labels': labels,
            'centers': centers,
            'inertia': kmeans.inertia_
        }
    
    elif algorithm == 'DBSCAN':
        dbscan = DBSCAN(eps=kwargs.get('eps', 0.5), 
                       min_samples=kwargs.get('min_samples', 5))
        labels = dbscan.fit_predict(X_scaled)
        unique_labels = np.unique(labels)
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(labels == -1)
        result = {
            'labels': labels,
            'n_clusters': n_clusters_found,
            'n_noise': n_noise,
            'eps': kwargs.get('eps', 0.5),
            'min_samples': kwargs.get('min_samples', 5)
        }
    
    return {
        'X': X,
        'X_scaled': X_scaled,
        'y_true': y_true,
        'labels': labels,
        'result': result
    }
'''