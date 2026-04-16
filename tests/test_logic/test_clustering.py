"""
Tests for clustering logic module.
"""

import pytest
import numpy as np
from app.logic.clustering import (
    generate_clustering_data,
    generate_complex_clustering_data,
    apply_kmeans,
    apply_dbscan,
    apply_hierarchical,
    apply_gmm,
    calculate_clustering_metrics,
    find_optimal_k,
    visualize_clusters_2d,
    visualize_clusters_3d,
)


class TestClusteringDataGeneration:
    """Test clustering data generation functions."""

    def test_generate_clustering_data_basic(self):
        """Test basic clustering data generation."""
        X, y_true = generate_clustering_data(
            n_samples=300,
            n_clusters=3,
            cluster_std=0.5,
            random_state=42,
        )

        assert X.shape == (300, 2)
        assert y_true.shape == (300,)
        assert len(np.unique(y_true)) == 3
        assert isinstance(X, np.ndarray)
        assert isinstance(y_true, np.ndarray)

    def test_generate_clustering_data_custom_params(self):
        """Test clustering data generation with custom parameters."""
        X, y_true = generate_clustering_data(
            n_samples=500,
            n_clusters=5,
            cluster_std=1.0,
            random_state=123,
        )

        assert X.shape == (500, 2)
        assert len(np.unique(y_true)) == 5

        # Check that data is shuffled
        assert not np.all(np.diff(y_true) == 0)

    def test_generate_complex_clustering_data_moons(self):
        """Test moons dataset generation."""
        X, y_true = generate_complex_clustering_data(
            n_samples=200,
            dataset_type="moons",
            noise=0.05,
            random_state=42,
        )

        assert X.shape == (200, 2)
        assert y_true.shape == (200,)
        assert len(np.unique(y_true)) == 2

    def test_generate_complex_clustering_data_circles(self):
        """Test circles dataset generation."""
        X, y_true = generate_complex_clustering_data(
            n_samples=200,
            dataset_type="circles",
            noise=0.1,
            random_state=42,
        )

        assert X.shape == (200, 2)
        assert len(np.unique(y_true)) == 2

    def test_generate_complex_clustering_data_anisotropic(self):
        """Test anisotropic dataset generation."""
        X, y_true = generate_complex_clustering_data(
            n_samples=200,
            dataset_type="anisotropic",
            noise=0.05,
            random_state=42,
        )

        assert X.shape == (200, 2)
        assert len(np.unique(y_true)) == 3  # Default is 3 clusters


class TestClusteringAlgorithms:
    """Test clustering algorithm applications."""

    @pytest.fixture
    def sample_data(self):
        """Sample data for clustering tests."""
        X, _ = generate_clustering_data(
            n_samples=100,
            n_clusters=3,
            cluster_std=0.3,
            random_state=42,
        )
        return X

    def test_apply_kmeans(self, sample_data):
        """Test K-means clustering."""
        result = apply_kmeans(
            sample_data,
            n_clusters=3,
            random_state=42,
            scale_data=True,
        )

        assert result["algorithm"] == "K-means"
        assert "labels" in result
        assert "centers" in result
        assert "inertia" in result
        assert "metrics" in result

        labels = result["labels"]
        assert len(labels) == len(sample_data)
        assert len(np.unique(labels)) == 3

        # Check metrics
        metrics = result["metrics"]
        assert "silhouette_score" in metrics
        assert isinstance(metrics["silhouette_score"], float)

    def test_apply_dbscan(self, sample_data):
        """Test DBSCAN clustering."""
        result = apply_dbscan(
            sample_data,
            eps=0.5,
            min_samples=5,
            scale_data=True,
        )

        assert result["algorithm"] == "DBSCAN"
        assert "labels" in result
        assert "eps" in result
        assert "min_samples" in result
        assert "n_clusters" in result
        assert "n_noise" in result

        labels = result["labels"]
        assert len(labels) == len(sample_data)

        # DBSCAN may find different number of clusters
        assert result["n_clusters"] >= 0
        assert result["n_noise"] >= 0

    def test_apply_hierarchical(self, sample_data):
        """Test hierarchical clustering."""
        result = apply_hierarchical(
            sample_data,
            n_clusters=3,
            linkage="ward",
            scale_data=True,
        )

        assert result["algorithm"] == "Hierarchical (ward)"
        assert "labels" in result
        assert "linkage" in result
        assert "n_clusters" in result
        assert "metrics" in result

        labels = result["labels"]
        assert len(labels) == len(sample_data)
        assert len(np.unique(labels)) == 3

    def test_apply_gmm(self, sample_data):
        """Test Gaussian Mixture Model clustering."""
        result = apply_gmm(
            sample_data,
            n_components=3,
            covariance_type="full",
            random_state=42,
            scale_data=True,
        )

        assert result["algorithm"] == "GMM (full)"
        assert "labels" in result
        assert "probabilities" in result
        assert "n_components" in result
        assert "covariance_type" in result
        assert "bic" in result
        assert "aic" in result
        assert "converged" in result

        labels = result["labels"]
        probabilities = result["probabilities"]

        assert len(labels) == len(sample_data)
        assert probabilities.shape == (len(sample_data), 3)
        assert len(np.unique(labels)) == 3


class TestClusteringMetrics:
    """Test clustering metrics calculation."""

    def test_calculate_clustering_metrics_valid(self):
        """Test metrics calculation with valid clusters."""
        X = np.random.randn(100, 2)
        labels = np.random.randint(0, 3, 100)  # 3 clusters

        metrics = calculate_clustering_metrics(X, labels)

        assert "silhouette_score" in metrics
        assert "calinski_harabasz_score" in metrics
        assert "davies_bouldin_score" in metrics

        # Check value ranges
        assert -1 <= metrics["silhouette_score"] <= 1
        assert metrics["calinski_harabasz_score"] >= 0
        assert metrics["davies_bouldin_score"] >= 0

    def test_calculate_clustering_metrics_single_cluster(self):
        """Test metrics calculation with single cluster."""
        X = np.random.randn(100, 2)
        labels = np.zeros(100)  # All points in same cluster

        metrics = calculate_clustering_metrics(X, labels)

        # Should return empty dict for single cluster
        assert metrics == {}

    def test_calculate_clustering_metrics_noise_only(self):
        """Test metrics calculation with only noise points."""
        X = np.random.randn(100, 2)
        labels = np.full(100, -1)  # All noise points

        metrics = calculate_clustering_metrics(X, labels)

        # Should return empty dict for noise only
        assert metrics == {}


class TestOptimalKAnalysis:
    """Test optimal k finding functions."""

    @pytest.fixture
    def sample_data_for_k(self):
        """Sample data for optimal k tests."""
        X, _ = generate_clustering_data(
            n_samples=200,
            n_clusters=4,
            cluster_std=0.5,
            random_state=42,
        )
        return X

    def test_find_optimal_k(self, sample_data_for_k):
        """Test optimal k finding."""
        result = find_optimal_k(
            sample_data_for_k,
            k_range=(2, 8),
            scale_data=True,
        )

        assert "results" in result
        assert "optimal_k_silhouette" in result
        assert "optimal_k_elbow" in result

        results = result["results"]
        assert len(results) == 7  # k from 2 to 8 inclusive

        # Check each k has results
        for k in range(2, 9):
            assert k in results
            assert "inertia" in results[k]
            assert "silhouette_score" in results[k]
            assert "labels" in results[k]
            assert "centers" in results[k]

        # Check optimal k values are in range
        assert 2 <= result["optimal_k_silhouette"] <= 8
        assert 2 <= result["optimal_k_elbow"] <= 8


class TestVisualization:
    """Test clustering visualization functions."""

    @pytest.fixture
    def sample_data_for_viz(self):
        """Sample data for visualization tests."""
        X, labels = generate_clustering_data(
            n_samples=50,
            n_clusters=3,
            cluster_std=0.3,
            random_state=42,
        )
        return X, labels

    def test_visualize_clusters_2d(self, sample_data_for_viz):
        """Test 2D cluster visualization."""
        X, labels = sample_data_for_viz
        centers = np.array([[0, 0], [1, 1], [2, 2]])

        fig = visualize_clusters_2d(
            X,
            labels,
            title="Test Clusters",
            centers=centers,
        )

        assert fig is not None
        assert hasattr(fig, "data")
        assert len(fig.data) > 0

        # Check title
        assert fig.layout.title.text == "Test Clusters"

    def test_visualize_clusters_2d_no_centers(self, sample_data_for_viz):
        """Test 2D cluster visualization without centers."""
        X, labels = sample_data_for_viz

        fig = visualize_clusters_2d(
            X,
            labels,
            title="Test Clusters No Centers",
        )

        assert fig is not None
        assert hasattr(fig, "data")

    def test_visualize_clusters_3d(self, sample_data_for_viz):
        """Test 3D cluster visualization."""
        X, labels = sample_data_for_viz

        # Add third dimension
        X_3d = np.hstack([X, np.random.randn(X.shape[0], 1)])

        fig = visualize_clusters_3d(
            X_3d,
            labels,
            title="Test 3D Clusters",
        )

        assert fig is not None
        assert hasattr(fig, "data")

        # Check it's a 3D plot
        assert hasattr(fig.layout, "scene")

    def test_visualize_clusters_3d_high_dim(self):
        """Test 3D visualization with high-dimensional data."""
        X = np.random.randn(50, 10)  # 10 dimensions
        labels = np.random.randint(0, 3, 50)

        fig = visualize_clusters_3d(
            X,
            labels,
            title="High-dim to 3D",
        )

        assert fig is not None
        # Should use PCA to reduce to 3D
        assert "PCA" in fig.layout.title.text


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_data(self):
        """Test with empty data."""
        X = np.array([]).reshape(0, 2)
        labels = np.array([])

        metrics = calculate_clustering_metrics(X, labels)
        assert metrics == {}

    def test_single_point(self):
        """Test with single data point."""
        X = np.array([[1, 2]])
        labels = np.array([0])

        metrics = calculate_clustering_metrics(X, labels)
        assert metrics == {}

    def test_apply_kmeans_invalid_k(self):
        """Test K-means with invalid k."""
        X = np.random.randn(10, 2)

        # k larger than samples
        with pytest.raises(ValueError):
            apply_kmeans(X, n_clusters=20)

    def test_scale_data_false(self):
        """Test clustering without scaling."""
        X, _ = generate_clustering_data(n_samples=50, n_clusters=2)

        result = apply_kmeans(X, n_clusters=2, scale_data=False)
        assert result["algorithm"] == "K-means"
        assert result["scaler"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
