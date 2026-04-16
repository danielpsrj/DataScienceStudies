#!/usr/bin/env python3
"""
Generate fake data for the Data Science Platform.
Populates the database with sample concepts, algorithms, datasets, and experiments.
"""

import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.data.repositories import (
    db,
    ConceptRepository,
    AlgorithmRepository,
    DatasetRepository,
    ExperimentRepository,
    UserRepository,
    SavedModelRepository,
    VisualizationRepository,
)
from app.data.models import Base


def create_concepts_and_algorithms() -> Dict[str, int]:
    """Create sample concepts and algorithms."""
    print("Creating concepts and algorithms...")

    concept_ids = {}

    with ConceptRepository() as concept_repo, AlgorithmRepository() as algo_repo:
        # Regression concept
        regression = concept_repo.create_concept(
            name="regression",
            display_name="Regression Analysis",
            category="supervised",
            description="Statistical methods for modeling relationships between variables and predicting continuous outcomes.",
            difficulty_level="beginner",
            mathematical_formulation="y = f(X) + ε",
            common_algorithms=[
                "linear_regression",
                "ridge_regression",
                "lasso_regression",
                "polynomial_regression",
            ],
            use_cases=[
                "price prediction",
                "demand forecasting",
                "risk assessment",
                "trend analysis",
            ],
            prerequisites=["basic statistics", "linear algebra"],
        )
        concept_ids["regression"] = regression.id

        # Create regression algorithms
        algo_repo.create_algorithm(
            name="linear_regression",
            display_name="Linear Regression",
            concept_id=regression.id,
            description="Models the linear relationship between independent variables and a dependent variable.",
            mathematical_formulation="y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε",
            parameters={
                "fit_intercept": {
                    "type": "bool",
                    "default": True,
                    "description": "Whether to calculate the intercept",
                },
                "normalize": {
                    "type": "bool",
                    "default": False,
                    "description": "Whether to normalize features",
                },
            },
            strengths=[
                "Simple to implement",
                "Interpretable coefficients",
                "Fast training",
            ],
            weaknesses=[
                "Assumes linearity",
                "Sensitive to outliers",
                "Multicollinearity issues",
            ],
            implementation_libraries=["scikit-learn", "statsmodels", "numpy"],
        )

        algo_repo.create_algorithm(
            name="ridge_regression",
            display_name="Ridge Regression (L2 Regularization)",
            concept_id=regression.id,
            description="Linear regression with L2 regularization to prevent overfitting.",
            mathematical_formulation="min ||y - Xβ||² + α||β||²",
            parameters={
                "alpha": {
                    "type": "float",
                    "default": 1.0,
                    "description": "Regularization strength",
                },
                "solver": {
                    "type": "str",
                    "default": "auto",
                    "description": "Solver to use",
                },
            },
            strengths=[
                "Reduces overfitting",
                "Handles multicollinearity",
                "Stable solutions",
            ],
            weaknesses=[
                "All coefficients remain non-zero",
                "Requires hyperparameter tuning",
            ],
            implementation_libraries=["scikit-learn", "statsmodels"],
        )

        # Clustering concept
        clustering = concept_repo.create_concept(
            name="clustering",
            display_name="Clustering Analysis",
            category="unsupervised",
            description="Grouping similar data points together based on their characteristics without prior labels.",
            difficulty_level="intermediate",
            mathematical_formulation="argmin_S Σᵢ=1ᵏ Σ_{x∈Sᵢ} ||x - μᵢ||²",
            common_algorithms=["kmeans", "dbscan", "hierarchical", "gaussian_mixture"],
            use_cases=[
                "customer segmentation",
                "anomaly detection",
                "image segmentation",
                "document clustering",
            ],
            prerequisites=["distance metrics", "dimensionality reduction"],
        )
        concept_ids["clustering"] = clustering.id

        # Create clustering algorithms
        algo_repo.create_algorithm(
            name="kmeans",
            display_name="K-Means Clustering",
            concept_id=clustering.id,
            description="Partitions data into k clusters by minimizing within-cluster variance.",
            mathematical_formulation="argmin_S Σᵢ=1ᵏ Σ_{x∈Sᵢ} ||x - μᵢ||²",
            parameters={
                "n_clusters": {
                    "type": "int",
                    "default": 8,
                    "description": "Number of clusters",
                },
                "max_iter": {
                    "type": "int",
                    "default": 300,
                    "description": "Maximum iterations",
                },
                "random_state": {
                    "type": "int",
                    "default": None,
                    "description": "Random seed",
                },
            },
            strengths=["Simple and fast", "Scales well", "Easy to interpret"],
            weaknesses=[
                "Requires specifying k",
                "Sensitive to initialization",
                "Assumes spherical clusters",
            ],
            implementation_libraries=["scikit-learn", "numpy"],
        )

        algo_repo.create_algorithm(
            name="dbscan",
            display_name="DBSCAN (Density-Based Clustering)",
            concept_id=clustering.id,
            description="Groups together points that are closely packed together, marking outliers as noise.",
            mathematical_formulation="Core point: |N_ε(p)| ≥ minPts",
            parameters={
                "eps": {
                    "type": "float",
                    "default": 0.5,
                    "description": "Maximum distance between points",
                },
                "min_samples": {
                    "type": "int",
                    "default": 5,
                    "description": "Minimum samples in neighborhood",
                },
            },
            strengths=[
                "No need to specify k",
                "Handles noise",
                "Finds arbitrary shapes",
            ],
            weaknesses=[
                "Sensitive to parameters",
                "Struggles with varying densities",
                "High-dimensional data",
            ],
            implementation_libraries=["scikit-learn"],
        )

        # Classification concept
        classification = concept_repo.create_concept(
            name="classification",
            display_name="Classification",
            category="supervised",
            description="Predicting categorical class labels based on input features.",
            difficulty_level="intermediate",
            mathematical_formulation="ŷ = argmax_c P(c|X)",
            common_algorithms=[
                "logistic_regression",
                "decision_tree",
                "random_forest",
                "svm",
            ],
            use_cases=[
                "spam detection",
                "image recognition",
                "fraud detection",
                "medical diagnosis",
            ],
            prerequisites=["probability theory", "linear algebra"],
        )
        concept_ids["classification"] = classification.id

        # Time Series concept
        time_series = concept_repo.create_concept(
            name="time_series",
            display_name="Time Series Analysis",
            category="supervised",
            description="Analyzing and forecasting data points collected over time intervals.",
            difficulty_level="advanced",
            mathematical_formulation="y_t = f(t, y_{t-1}, y_{t-2}, ...) + ε_t",
            common_algorithms=["arima", "exponential_smoothing", "prophet", "lstm"],
            use_cases=[
                "stock prediction",
                "weather forecasting",
                "sales forecasting",
                "energy demand",
            ],
            prerequisites=["statistics", "signal processing"],
        )
        concept_ids["time_series"] = time_series.id

    print(f"Created concepts: {list(concept_ids.keys())}")
    return concept_ids


def create_sample_datasets() -> Dict[str, int]:
    """Create sample dataset records."""
    print("Creating sample datasets...")

    dataset_ids = {}

    with DatasetRepository() as dataset_repo:
        # Synthetic regression dataset
        regression_dataset = dataset_repo.create_dataset(
            name="synthetic_linear_data",
            description="Synthetic linear regression dataset with noise",
            dataset_type="synthetic",
            source="generated",
            columns=["feature_1", "feature_2", "target"],
            num_rows=1000,
            num_features=2,
            target_column="target",
            is_public=True,
        )
        dataset_ids["regression"] = regression_dataset.id

        # Synthetic clustering dataset
        clustering_dataset = dataset_repo.create_dataset(
            name="synthetic_clusters",
            description="Synthetic clustering dataset with 4 Gaussian blobs",
            dataset_type="synthetic",
            source="generated",
            columns=["x", "y", "cluster"],
            num_rows=500,
            num_features=2,
            target_column="cluster",
            is_public=True,
        )
        dataset_ids["clustering"] = clustering_dataset.id

        # Classification dataset (Iris-like)
        classification_dataset = dataset_repo.create_dataset(
            name="synthetic_flowers",
            description="Synthetic flower classification dataset",
            dataset_type="synthetic",
            source="generated",
            columns=[
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
                "species",
            ],
            num_rows=150,
            num_features=4,
            target_column="species",
            is_public=True,
        )
        dataset_ids["classification"] = classification_dataset.id

        # Time series dataset
        time_series_dataset = dataset_repo.create_dataset(
            name="synthetic_sales",
            description="Synthetic monthly sales data with trend and seasonality",
            dataset_type="synthetic",
            source="generated",
            columns=["date", "sales", "promotion", "holiday"],
            num_rows=365,
            num_features=3,
            target_column="sales",
            is_public=True,
        )
        dataset_ids["time_series"] = time_series_dataset.id

    print(f"Created datasets: {list(dataset_ids.keys())}")
    return dataset_ids


def create_sample_users() -> Dict[str, int]:
    """Create sample users."""
    print("Creating sample users...")

    user_ids = {}

    with UserRepository() as user_repo:
        # Demo user
        demo_user = user_repo.create_user(
            username="demo_user",
            email="demo@datascienceplatform.com",
        )
        user_ids["demo"] = demo_user.id

        # Researcher user
        researcher = user_repo.create_user(
            username="researcher",
            email="researcher@university.edu",
        )
        user_ids["researcher"] = researcher.id

        # Student user
        student = user_repo.create_user(
            username="student",
            email="student@learning.org",
        )
        user_ids["student"] = student.id

    print(f"Created users: {list(user_ids.keys())}")
    return user_ids


def create_sample_experiments(
    user_ids: Dict[str, int], concept_ids: Dict[str, int]
) -> None:
    """Create sample experiments."""
    print("Creating sample experiments...")

    with ExperimentRepository() as exp_repo:
        # Regression experiment
        exp_repo.create_experiment(
            user_id=user_ids["demo"],
            name="House Price Prediction",
            description="Predicting house prices based on features like size, location, and age",
            concept_type="regression",
            algorithm="linear_regression",
            parameters={
                "n_samples": 1000,
                "n_features": 5,
                "test_size": 0.2,
                "random_state": 42,
            },
            metrics={
                "mse": 0.045,
                "r2": 0.892,
                "mae": 0.167,
                "rmse": 0.212,
            },
            dataset_info={
                "name": "synthetic_linear_data",
                "n_rows": 1000,
                "n_features": 5,
                "target": "price",
            },
            is_public=True,
        )

        # Clustering experiment
        exp_repo.create_experiment(
            user_id=user_ids["researcher"],
            name="Customer Segmentation",
            description="Segmenting customers based on purchasing behavior and demographics",
            concept_type="clustering",
            algorithm="kmeans",
            parameters={
                "n_clusters": 4,
                "max_iter": 300,
                "random_state": 42,
                "n_init": 10,
            },
            metrics={
                "silhouette_score": 0.712,
                "calinski_harabasz": 456.3,
                "davies_bouldin": 0.823,
                "inertia": 1234.5,
            },
            dataset_info={
                "name": "synthetic_clusters",
                "n_rows": 500,
                "n_features": 2,
                "target": "cluster",
            },
            is_public=True,
        )

        # Classification experiment
        exp_repo.create_experiment(
            user_id=user_ids["student"],
            name="Flower Species Classification",
            description="Classifying flower species based on morphological measurements",
            concept_type="classification",
            algorithm="random_forest",
            parameters={
                "n_estimators": 100,
                "max_depth": 10,
                "random_state": 42,
                "criterion": "gini",
            },
            metrics={
                "accuracy": 0.967,
                "precision": 0.972,
                "recall": 0.965,
                "f1_score": 0.968,
            },
            dataset_info={
                "name": "synthetic_flowers",
                "n_rows": 150,
                "n_features": 4,
                "target": "species",
            },
            is_public=True,
        )

        # Time series experiment
        exp_repo.create_experiment(
            user_id=user_ids["demo"],
            name="Sales Forecasting",
            description="Forecasting monthly sales with trend and seasonality",
            concept_type="time_series",
            algorithm="arima",
            parameters={
                "order": "(1,1,1)",
                "seasonal_order": "(1,1,1,12)",
                "trend": "c",
            },
            metrics={
                "mae": 12.45,
                "mse": 234.67,
                "rmse": 15.32,
                "mape": 0.082,
            },
            dataset_info={
                "name": "synthetic_sales",
                "n_rows": 365,
                "n_features": 3,
                "target": "sales",
            },
            is_public=True,
        )

    print("Created sample experiments")


def create_sample_models(user_ids: Dict[str, int]) -> None:
    """Create sample saved models."""
    print("Creating sample saved models...")

    with SavedModelRepository() as model_repo:
        # Linear regression model
        model_repo.save_model(
            user_id=user_ids["demo"],
            name="House Price Predictor v1",
            model_type="regression",
            algorithm="linear_regression",
            model_data={
                "coefficients": [2.5, 1.8, -0.3, 0.7, 0.1],
                "intercept": 150.0,
                "feature_names": ["size", "bedrooms", "age", "location", "condition"],
            },
            metrics={
                "mse": 0.045,
                "r2": 0.892,
                "training_time": 0.12,
            },
            feature_names=["size", "bedrooms", "age", "location", "condition"],
            target_name="price",
            is_public=True,
        )

        # K-means model
        model_repo.save_model(
            user_id=user_ids["researcher"],
            name="Customer Segments v2",
            model_type="clustering",
            algorithm="kmeans",
            model_data={
                "cluster_centers": [[1.2, 2.3], [3.4, 4.5], [5.6, 6.7], [7.8, 8.9]],
                "n_clusters": 4,
                "inertia": 1234.5,
            },
            metrics={
                "silhouette_score": 0.712,
                "calinski_harabasz": 456.3,
                "training_time": 0.08,
            },
            feature_names=["spending", "frequency"],
            is_public=True,
        )

        # Random forest model
        model_repo.save_model(
            user_id=user_ids["student"],
            name="Flower Classifier",
            model_type="classification",
            algorithm="random_forest",
            model_data={
                "n_estimators": 100,
                "max_depth": 10,
                "feature_importances": [0.3, 0.25, 0.35, 0.1],
                "classes": ["setosa", "versicolor", "virginica"],
            },
            metrics={
                "accuracy": 0.967,
                "precision": 0.972,
                "training_time": 0.45,
            },
            feature_names=[
                "sepal_length",
                "sepal_width",
                "petal_length",
                "petal_width",
            ],
            target_name="species",
            is_public=True,
        )

    print("Created sample saved models")


def main() -> None:
    """Main function to generate all fake data."""
    print("=" * 60)
    print("Generating fake data for Data Science Platform")
    print("=" * 60)

    try:
        # Create database tables
        print("\n1. Creating database tables...")
        db.create_tables()

        # Create concepts and algorithms
        concept_ids = create_concepts_and_algorithms()

        # Create datasets
        dataset_ids = create_sample_datasets()

        # Create users
        user_ids = create_sample_users()

        # Create experiments
        create_sample_experiments(user_ids, concept_ids)

        # Create saved models
        create_sample_models(user_ids)

        print("\n" + "=" * 60)
        print("Fake data generation completed successfully!")
        print("=" * 60)
        print("\nSummary:")
        print(f"- Concepts created: {len(concept_ids)}")
        print(f"- Datasets created: {len(dataset_ids)}")
        print(f"- Users created: {len(user_ids)}")
        print(f"- Database: {db.database_url}")

    except Exception as e:
        print(f"\nError generating fake data: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
