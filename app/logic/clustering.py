"""
Clustering logic and utilities.
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, List, Dict, Any, Optional

from app.caching import cache_data, get_cached_data


def generate_clustering_data(
    n_samples: int = 300,
    n_clusters: int = 3,
    cluster_std: float = 0.5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic clustering data with Gaussian blobs.
    
    Args:
        n_samples: Total number of samples
        n_clusters: Number of clusters to generate
        cluster_std: Standard deviation of clusters
        random_state: Random seed
        
    Returns:
        Tuple of (X, y_true) arrays where y_true are true cluster labels
    """
    np.random.seed(random_state)
    
    # Generate cluster centers
    centers = np.random.randn(n_clusters, 2) * 3
    
    # Generate samples per cluster
    samples_per_cluster = n_samples // n_clusters
    X_list = []
    y_list = []
    
    for cluster_id in range(n_clusters):
        # Generate samples for this cluster
        cluster_samples = np.random.randn(samples_per_cluster, 2) * cluster_std + centers[cluster_id]
        X_list.append(cluster_samples)
        y_list.append(np.full(samples_per_cluster, cluster_id))
    
    # Handle remainder samples
    remainder = n_samples % n_clusters
    if remainder > 0:
        cluster_id = np.random.randint(0, n_clusters)
        remainder_samples = np.random.randn(remainder, 2) * cluster_std + centers[cluster_id]
        X_list.append(remainder_samples)
        y_list.append(np.full(remainder, cluster_id))
    
    # Combine all samples
    X = np.vstack(X_list)
    y_true = np.hstack(y_list)
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y_true = y_true[shuffle_idx]
    
    return X, y_true


def generate_complex_clustering_data(
    n_samples: int = 500,
    dataset_type: str = "moons",
    noise: float = 0.05,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate complex clustering datasets (moons, circles, blobs).
    
    Args:
        n_samples: Number of samples
        dataset_type: Type of dataset ("moons", "circles", "blobs", "anisotropic")
        noise: Noise level for the dataset
        random_state: Random seed
        
    Returns:
        Tuple of (X, y_true) arrays
    """
    from sklearn.datasets import make_moons, make_circles, make_blobs
    
    np.random.seed(random_state)
    
    if dataset_type == "moons":
        X, y_true = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif dataset_type == "circles":
        X, y_true = make_circles(n_samples=n_samples, noise=noise, random_state=random_state, factor=0.5)
    elif dataset_type == "anisotropic":
        # Generate anisotropic blobs
        X, y_true = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.6, -0.6], [-0.4, 0.8]]
        X = np.dot(X, transformation)
    else:  # "blobs"
        X, y_true = make_blobs(n_samples=n_samples, centers=4, random_state=random_state)
    
    return X, y_true


def apply_kmeans(
    X: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    scale_data: bool = True,
) -> Dict[str, Any]:
    """
    Apply K-means clustering to data.
    
    Args:
        X: Input data
        n_clusters: Number of clusters
        random_state: Random seed
        scale_data: Whether to scale data before clustering
        
    Returns:
        Dictionary with clustering results and metrics
    """
    # Scale data if requested
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Apply K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    centers = kmeans.cluster_centers_
    
    # Calculate metrics
    metrics = calculate_clustering_metrics(X_scaled, labels)
    
    # Inverse transform centers if data was scaled
    if scale_data:
        centers = scaler.inverse_transform(centers)
    
    return {
        "algorithm": "K-means",
        "labels": labels,
        "centers": centers,
        "inertia": kmeans.inertia_,
        "n_iter": kmeans.n_iter_,
        "metrics": metrics,
        "scaler": scaler if scale_data else None,
    }


def apply_dbscan(
    X: np.ndarray,
    eps: float = 0.5,
    min_samples: int = 5,
    scale_data: bool = True,
) -> Dict[str, Any]:
    """
    Apply DBSCAN clustering to data.
    
    Args:
        X: Input data
        eps: Maximum distance between samples in the same neighborhood
        min_samples: Minimum number of samples in a neighborhood
        scale_data: Whether to scale data before clustering
        
    Returns:
        Dictionary with clustering results and metrics
    """
    # Scale data if requested
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)
    
    # Count clusters (excluding noise labeled as -1)
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    
    # Calculate metrics (only if we have at least 2 clusters)
    metrics = {}
    if n_clusters >= 2:
        # Filter out noise points for metric calculation
        non_noise_mask = labels != -1
        if np.sum(non_noise_mask) > 0:
            metrics = calculate_clustering_metrics(
                X_scaled[non_noise_mask], 
                labels[non_noise_mask]
            )
    
    return {
        "algorithm": "DBSCAN",
        "labels": labels,
        "eps": eps,
        "min_samples": min_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "metrics": metrics,
        "scaler": scaler if scale_data else None,
    }


def apply_hierarchical(
    X: np.ndarray,
    n_clusters: int = 3,
    linkage: str = "ward",
    scale_data: bool = True,
) -> Dict[str, Any]:
    """
    Apply hierarchical clustering to data.
    
    Args:
        X: Input data
        n_clusters: Number of clusters
        linkage: Linkage criterion ("ward", "complete", "average", "single")
        scale_data: Whether to scale data before clustering
        
    Returns:
        Dictionary with clustering results and metrics
    """
    # Scale data if requested
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Apply hierarchical clustering
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage=linkage
    )
    labels = hierarchical.fit_predict(X_scaled)
    
    # Calculate metrics
    metrics = calculate_clustering_metrics(X_scaled, labels)
    
    return {
        "algorithm": f"Hierarchical ({linkage})",
        "labels": labels,
        "linkage": linkage,
        "n_clusters": n_clusters,
        "metrics": metrics,
        "scaler": scaler if scale_data else None,
    }


def apply_gmm(
    X: np.ndarray,
    n_components: int = 3,
    covariance_type: str = "full",
    random_state: int = 42,
    scale_data: bool = True,
) -> Dict[str, Any]:
    """
    Apply Gaussian Mixture Model clustering to data.
    
    Args:
        X: Input data
        n_components: Number of mixture components
        covariance_type: Type of covariance parameters
        random_state: Random seed
        scale_data: Whether to scale data before clustering
        
    Returns:
        Dictionary with clustering results and metrics
    """
    # Scale data if requested
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # Apply GMM
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state
    )
    labels = gmm.fit_predict(X_scaled)
    probabilities = gmm.predict_proba(X_scaled)
    
    # Calculate metrics
    metrics = calculate_clustering_metrics(X_scaled, labels)
    
    # Calculate BIC and AIC
    bic = gmm.bic(X_scaled)
    aic = gmm.aic(X_scaled)
    
    return {
        "algorithm": f"GMM ({covariance_type})",
        "labels": labels,
        "probabilities": probabilities,
        "n_components": n_components,
        "covariance_type": covariance_type,
        "bic": bic,
        "aic": aic,
        "converged": gmm.converged_,
        "n_iter": gmm.n_iter_,
        "metrics": metrics,
        "scaler": scaler if scale_data else None,
    }


def calculate_clustering_metrics(
    X: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate clustering validation metrics.
    
    Args:
        X: Input data
        labels: Cluster labels
        
    Returns:
        Dictionary with metric names and values
    """
    n_clusters = len(set(labels))
    
    # Need at least 2 clusters and more than 1 sample per cluster for metrics
    if n_clusters < 2 or len(X) < 2:
        return {}
    
    metrics = {}
    
    try:
        metrics["silhouette_score"] = silhouette_score(X, labels)
    except:
        metrics["silhouette_score"] = np.nan
    
    try:
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(X, labels)
    except:
        metrics["calinski_harabasz_score"] = np.nan
    
    try:
        metrics["davies_bouldin_score"] = davies_bouldin_score(X, labels)
    except:
        metrics["davies_bouldin_score"] = np.nan
    
    return metrics


def find_optimal_k(
    X: np.ndarray,
    k_range: Tuple[int, int] = (2, 10),
    scale_data: bool = True,
) -> Dict[str, Any]:
    """
    Find optimal number of clusters using elbow method and silhouette analysis.
    
    Args:
        X: Input data
        k_range: Range of k values to test (min, max)
        scale_data: Whether to scale data before clustering
        
    Returns:
        Dictionary with results for each k value
    """
    # Scale data if requested
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    k_min, k_max = k_range
    results = {}
    
    for k in range(k_min, k_max + 1):
        # Apply K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        inertia = kmeans.inertia_
        
        # Calculate silhouette score (only if k > 1)
        silhouette = np.nan
        if k > 1:
            try:
                silhouette = silhouette_score(X_scaled, labels)
            except:
                pass
        
        results[k] = {
            "inertia": inertia,
            "silhouette_score": silhouette,
            "labels": labels,
            "centers": kmeans.cluster_centers_,
        }
    
    # Find optimal k based on silhouette score
    silhouette_scores = [results[k]["silhouette_score"] for k in results]
    valid_scores = [s for s in silhouette_scores if not np.isnan(s)]
    
    if valid_scores:
        optimal_k_silhouette = list(results.keys())[
            silhouette_scores.index(max(valid_scores))
        ]
    else:
        optimal_k_silhouette = k_min
    
    # Find elbow point (knee point)
    inertias = [results[k]["inertia"] for k in results]
    optimal_k_elbow = _find_elbow_point(list(results.keys()), inertias)
    
    return {
        "results": results,
        "optimal_k_silhouette": optimal_k_silhouette,
        "optimal_k_elbow": optimal_k_elbow,
        "scaler": scaler if scale_data else None,
    }


def _find_elbow_point(k_values: List[int], inertias: List[float]) -> int:
    """
    Find elbow point using the knee point detection method.
    
    Args:
        k_values: List of k values
        inertias: List of inertia values
        
    Returns:
        Optimal k value at the elbow point
    """
    if len(k_values) < 3:
        return k_values[0] if k_values else 2
    
    # Normalize values
    k_norm = np.array(k_values) / max(k_values)
    inertia_norm = np.array(inertias) / max(inertias)
    
    # Calculate distances from line connecting first and last points
    line_start = np.array([k_norm[0], inertia_norm[0]])
    line_end = np.array([k_norm[-1], inertia_norm[-1]])
    line_vec = line_end - line_start
    line_len = np.linalg.norm(line_vec)
    
    if line_len == 0:
        return k_values[0]
    
    # Unit vector along the line
    line_unit = line_vec / line_len
    
    max_distance = -1
    elbow_idx = 0
    
    for i in range(1, len(k_values) - 1):
        point = np.array([k_norm[i], inertia_norm[i]])
        
        # Vector from line start to point
        point_vec = point - line_start
        
        # Project point onto line
        projection_length = np.dot(point_vec, line_unit)
        projection = line_start + projection_length * line_unit
        
        # Distance from point to line
        distance = np.linalg.norm(point - projection)
        
        if distance > max_distance:
            max_distance = distance
            elbow_idx = i
    
    return k_values[elbow_idx]


def visualize_clusters_2d(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "Cluster Visualization",
    centers: Optional[np.ndarray] = None,
) -> go.Figure:
    """
    Create 2D scatter plot of clusters.
    
    Args:
        X: 2D data points
        labels: Cluster labels
        title: Plot title
        centers: Cluster centers to plot
        
    Returns:
        Plotly figure object
    """
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        'x': X[:, 0],
        'y': X[:, 1],
        'cluster': labels.astype(str)
    })
    
    # Create scatter plot
    fig = px.scatter(
        df, x='x', y='y', color='cluster',
        title=title,
        labels={'cluster': 'Cluster'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # Add cluster centers if provided
    if centers is not None:
        fig.add_trace(go.Scatter(
            x=centers[:, 0],
            y=centers[:, 1],
            mode='markers',
            marker=dict(
                symbol='x',
                size=15,
                color='black',
                line=dict(width=2)
            ),
            name='Cluster Centers'
        ))
    
    fig.update_layout(
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        showlegend=True,
        hovermode='closest'
    )
    
    return fig


# Patch to complete the visualize_clusters_3d function

def visualize_clusters_3d(
    X: np.ndarray,
    labels: np.ndarray,
    title: str = "3D Cluster Visualization",
) -> go.Figure:
    """
    Create 3D scatter plot of clusters.
    
    Args:
        X: Data points (will use first 3 dimensions)
        labels: Cluster labels
        title: Plot title
        
    Returns:
        Plotly figure object
    """
    # Use PCA to reduce to 3D if needed
    if X.shape[1] > 3:
        pca = PCA(n_components=3)
        X_3d = pca.fit_transform(X)
        var_explained = pca.explained_variance_ratio_.sum()
        title = f"{title} (PCA: {var_explained:.1%} variance explained)"
    else:
        X_3d = X[:, :3]
        if X.shape[1] < 3:
            # Pad with zeros if we have fewer than 3 dimensions
            padding = np.zeros((X.shape[0], 3 - X.shape[1]))
            X_3d = np.hstack([X_3d, padding])
    
    # Convert to DataFrame for easier plotting
    df = pd.DataFrame({
        'x': X_3d[:, 0],
        'y': X_3d[:, 1],
        'z': X_3d[:, 2],
        'cluster': labels.astype(str)
    })
    
    # Create 3D scatter plot
    fig = px.scatter_3d(
        df, x='x', y='y', z='z', color='cluster',
        title=title,
        labels={'cluster': 'Cluster'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            zaxis_title="Dimension 3"
        ),
        showlegend=True,
        hovermode='closest'
    )
    
    return fig


# Example usage and testing
if __name__ == "__main__":
    # Generate sample data
    X, y_true = generate_clustering_data(n_samples=300, n_clusters=4)
    
    # Test K-means
    kmeans_result = apply_kmeans(X, n_clusters=4)
    print(f"K-means inertia: {kmeans_result['inertia']}")
    print(f"K-means metrics: {kmeans_result['metrics']}")
    
    # Test optimal k finding
    optimal_k_result = find_optimal_k(X, k_range=(2, 10))
    print(f"Optimal k (silhouette): {optimal_k_result['optimal_k_silhouette']}")
    print(f"Optimal k (elbow): {optimal_k_result['optimal_k_elbow']}")
    
    print("Clustering module loaded successfully!")