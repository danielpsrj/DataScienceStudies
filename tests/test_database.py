"""
Tests for database repositories.
"""

import pytest
from datetime import datetime
from typing import Dict, Any


class TestUserRepository:
    """Test UserRepository operations."""

    def test_create_user(self, test_db, sample_user_data):
        """Test creating a user."""
        user_repo = test_db["users"]

        user = user_repo.create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
        )

        assert user.id is not None
        assert user.username == sample_user_data["username"]
        assert user.email == sample_user_data["email"]
        assert user.created_at is not None
        assert user.is_active is True

    def test_get_user(self, test_db, sample_user_data):
        """Test getting a user by ID."""
        user_repo = test_db["users"]

        # Create user first
        user = user_repo.create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
        )

        # Retrieve user
        retrieved_user = user_repo.get_user(user.id)

        assert retrieved_user is not None
        assert retrieved_user.id == user.id
        assert retrieved_user.username == user.username
        assert retrieved_user.email == user.email

    def test_get_user_by_username(self, test_db, sample_user_data):
        """Test getting a user by username."""
        user_repo = test_db["users"]

        # Create user first
        user = user_repo.create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
        )

        # Retrieve by username
        retrieved_user = user_repo.get_user_by_username(sample_user_data["username"])

        assert retrieved_user is not None
        assert retrieved_user.username == sample_user_data["username"]

    def test_update_user_last_login(self, test_db, sample_user_data):
        """Test updating user's last login."""
        user_repo = test_db["users"]

        # Create user
        user = user_repo.create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
        )

        # Update last login
        updated_user = user_repo.update_user_last_login(user.id)

        assert updated_user is not None
        assert updated_user.last_login is not None
        assert isinstance(updated_user.last_login, datetime)

    def test_list_users(self, test_db):
        """Test listing users."""
        user_repo = test_db["users"]

        # Create multiple users
        for i in range(5):
            user_repo.create_user(
                username=f"test_user_{i}",
                email=f"test_{i}@example.com",
            )

        # List users
        users = user_repo.list_users(skip=0, limit=10)

        assert len(users) >= 5
        assert all(user.username.startswith("test_user_") for user in users[:5])


class TestExperimentRepository:
    """Test ExperimentRepository operations."""

    def test_create_experiment(self, test_db, sample_experiment_data, sample_user_data):
        """Test creating an experiment."""
        user_repo = test_db["users"]
        exp_repo = test_db["experiments"]

        # Create user first
        user = user_repo.create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
        )

        # Create experiment
        experiment = exp_repo.create_experiment(
            user_id=user.id,
            **sample_experiment_data,
        )

        assert experiment.id is not None
        assert experiment.name == sample_experiment_data["name"]
        assert experiment.concept_type == sample_experiment_data["concept_type"]
        assert experiment.algorithm == sample_experiment_data["algorithm"]
        assert experiment.parameters == sample_experiment_data["parameters"]
        assert experiment.metrics == sample_experiment_data["metrics"]
        assert experiment.dataset_info == sample_experiment_data["dataset_info"]
        assert experiment.is_public == sample_experiment_data["is_public"]
        assert experiment.created_at is not None
        assert experiment.updated_at is not None

    def test_get_experiment(self, test_db, sample_experiment_data, sample_user_data):
        """Test getting an experiment by ID."""
        user_repo = test_db["users"]
        exp_repo = test_db["experiments"]

        # Create user and experiment
        user = user_repo.create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
        )

        experiment = exp_repo.create_experiment(
            user_id=user.id,
            **sample_experiment_data,
        )

        # Retrieve experiment
        retrieved_experiment = exp_repo.get_experiment(experiment.id)

        assert retrieved_experiment is not None
        assert retrieved_experiment.id == experiment.id
        assert retrieved_experiment.name == experiment.name

    def test_update_experiment_metrics(
        self, test_db, sample_experiment_data, sample_user_data
    ):
        """Test updating experiment metrics."""
        user_repo = test_db["users"]
        exp_repo = test_db["experiments"]

        # Create user and experiment
        user = user_repo.create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
        )

        experiment = exp_repo.create_experiment(
            user_id=user.id,
            **sample_experiment_data,
        )

        # Update metrics
        new_metrics = {"new_metric": 0.99, "accuracy": 0.95}
        updated_experiment = exp_repo.update_experiment_metrics(
            experiment.id,
            new_metrics,
        )

        assert updated_experiment is not None
        assert "new_metric" in updated_experiment.metrics
        assert updated_experiment.metrics["new_metric"] == 0.99
        assert updated_experiment.metrics["accuracy"] == 0.95
        # Original metrics should still be there
        assert "mse" in updated_experiment.metrics

    def test_list_experiments(self, test_db, sample_experiment_data, sample_user_data):
        """Test listing experiments."""
        user_repo = test_db["users"]
        exp_repo = test_db["experiments"]

        # Create user
        user = user_repo.create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
        )

        # Create multiple experiments
        for i in range(3):
            exp_data = sample_experiment_data.copy()
            exp_data["name"] = f"{exp_data['name']} {i}"
            exp_repo.create_experiment(
                user_id=user.id,
                **exp_data,
            )

        # List experiments
        experiments = exp_repo.list_experiments(
            user_id=user.id,
            skip=0,
            limit=10,
        )

        assert len(experiments) == 3
        assert all(exp.name.startswith("Test Experiment") for exp in experiments)

    def test_delete_experiment(self, test_db, sample_experiment_data, sample_user_data):
        """Test deleting an experiment."""
        user_repo = test_db["users"]
        exp_repo = test_db["experiments"]

        # Create user and experiment
        user = user_repo.create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
        )

        experiment = exp_repo.create_experiment(
            user_id=user.id,
            **sample_experiment_data,
        )

        # Delete experiment
        deleted = exp_repo.delete_experiment(experiment.id)

        assert deleted is True

        # Try to retrieve deleted experiment
        retrieved_experiment = exp_repo.get_experiment(experiment.id)
        assert retrieved_experiment is None


class TestDatasetRepository:
    """Test DatasetRepository operations."""

    def test_create_dataset(self, test_db, sample_dataset_data):
        """Test creating a dataset."""
        dataset_repo = test_db["datasets"]

        dataset = dataset_repo.create_dataset(**sample_dataset_data)

        assert dataset.id is not None
        assert dataset.name == sample_dataset_data["name"]
        assert dataset.description == sample_dataset_data["description"]
        assert dataset.dataset_type == sample_dataset_data["dataset_type"]
        assert dataset.columns == sample_dataset_data["columns"]
        assert dataset.num_rows == sample_dataset_data["num_rows"]
        assert dataset.num_features == sample_dataset_data["num_features"]
        assert dataset.target_column == sample_dataset_data["target_column"]
        assert dataset.is_public == sample_dataset_data["is_public"]

    def test_get_dataset(self, test_db, sample_dataset_data):
        """Test getting a dataset by ID."""
        dataset_repo = test_db["datasets"]

        dataset = dataset_repo.create_dataset(**sample_dataset_data)

        retrieved_dataset = dataset_repo.get_dataset(dataset.id)

        assert retrieved_dataset is not None
        assert retrieved_dataset.id == dataset.id
        assert retrieved_dataset.name == dataset.name

    def test_get_dataset_by_name(self, test_db, sample_dataset_data):
        """Test getting a dataset by name."""
        dataset_repo = test_db["datasets"]

        dataset = dataset_repo.create_dataset(**sample_dataset_data)

        retrieved_dataset = dataset_repo.get_dataset_by_name(
            sample_dataset_data["name"]
        )

        assert retrieved_dataset is not None
        assert retrieved_dataset.name == sample_dataset_data["name"]

    def test_list_datasets(self, test_db, sample_dataset_data):
        """Test listing datasets."""
        dataset_repo = test_db["datasets"]

        # Create multiple datasets
        for i in range(3):
            data = sample_dataset_data.copy()
            data["name"] = f"{data['name']}_{i}"
            dataset_repo.create_dataset(**data)

        datasets = dataset_repo.list_datasets(skip=0, limit=10)

        assert len(datasets) == 3
        assert all(ds.name.startswith("test_dataset") for ds in datasets)


class TestConceptRepository:
    """Test ConceptRepository operations."""

    def test_create_concept(self, test_db, sample_concept_data):
        """Test creating a concept."""
        concept_repo = test_db["concepts"]

        concept = concept_repo.create_concept(**sample_concept_data)

        assert concept.id is not None
        assert concept.name == sample_concept_data["name"]
        assert concept.display_name == sample_concept_data["display_name"]
        assert concept.category == sample_concept_data["category"]
        assert concept.description == sample_concept_data["description"]
        assert concept.difficulty_level == sample_concept_data["difficulty_level"]
        assert concept.common_algorithms == sample_concept_data["common_algorithms"]
        assert concept.use_cases == sample_concept_data["use_cases"]
        assert concept.prerequisites == sample_concept_data["prerequisites"]

    def test_get_concept(self, test_db, sample_concept_data):
        """Test getting a concept by ID."""
        concept_repo = test_db["concepts"]

        concept = concept_repo.create_concept(**sample_concept_data)

        retrieved_concept = concept_repo.get_concept(concept.id)

        assert retrieved_concept is not None
        assert retrieved_concept.id == concept.id
        assert retrieved_concept.name == concept.name

    def test_get_concept_by_name(self, test_db, sample_concept_data):
        """Test getting a concept by name."""
        concept_repo = test_db["concepts"]

        concept = concept_repo.create_concept(**sample_concept_data)

        retrieved_concept = concept_repo.get_concept_by_name(
            sample_concept_data["name"]
        )

        assert retrieved_concept is not None
        assert retrieved_concept.name == sample_concept_data["name"]

    def test_list_concepts(self, test_db, sample_concept_data):
        """Test listing concepts."""
        concept_repo = test_db["concepts"]

        # Create multiple concepts
        for i in range(3):
            data = sample_concept_data.copy()
            data["name"] = f"{data['name']}_{i}"
            data["display_name"] = f"{data['display_name']} {i}"
            concept_repo.create_concept(**data)

        concepts = concept_repo.list_concepts(skip=0, limit=10)

        assert len(concepts) == 3
        assert all(concept.name.startswith("test_concept") for concept in concepts)


class TestAlgorithmRepository:
    """Test AlgorithmRepository operations."""

    def test_create_algorithm(
        self, test_db, sample_algorithm_data, sample_concept_data
    ):
        """Test creating an algorithm."""
        concept_repo = test_db["concepts"]
        algo_repo = test_db["algorithms"]

        # Create concept first
        concept = concept_repo.create_concept(**sample_concept_data)

        # Create algorithm
        algorithm = algo_repo.create_algorithm(
            concept_id=concept.id,
            **sample_algorithm_data,
        )

        assert algorithm.id is not None
        assert algorithm.name == sample_algorithm_data["name"]
        assert algorithm.display_name == sample_algorithm_data["display_name"]
        assert algorithm.concept_id == concept.id
        assert algorithm.description == sample_algorithm_data["description"]
        assert algorithm.parameters == sample_algorithm_data["parameters"]
        assert algorithm.strengths == sample_algorithm_data["strengths"]
        assert algorithm.weaknesses == sample_algorithm_data["weaknesses"]
        assert (
            algorithm.implementation_libraries
            == sample_algorithm_data["implementation_libraries"]
        )

    def test_get_algorithm(self, test_db, sample_algorithm_data, sample_concept_data):
        """Test getting an algorithm by ID."""
        concept_repo = test_db["concepts"]
        algo_repo = test_db["algorithms"]

        # Create concept and algorithm
        concept = concept_repo.create_concept(**sample_concept_data)
        algorithm = algo_repo.create_algorithm(
            concept_id=concept.id,
            **sample_algorithm_data,
        )

        retrieved_algorithm = algo_repo.get_algorithm(algorithm.id)

        assert retrieved_algorithm is not None
        assert retrieved_algorithm.id == algorithm.id
        assert retrieved_algorithm.name == algorithm.name

    def test_get_algorithm_by_name(
        self, test_db, sample_algorithm_data, sample_concept_data
    ):
        """Test getting an algorithm by name."""
        concept_repo = test_db["concepts"]
        algo_repo = test_db["algorithms"]

        # Create concept and algorithm
        concept = concept_repo.create_concept(**sample_concept_data)
        algorithm = algo_repo.create_algorithm(
            concept_id=concept.id,
            **sample_algorithm_data,
        )

        retrieved_algorithm = algo_repo.get_algorithm_by_name(
            sample_algorithm_data["name"]
        )

        assert retrieved_algorithm is not None
        assert retrieved_algorithm.name == sample_algorithm_data["name"]

    def test_list_algorithms(self, test_db, sample_algorithm_data, sample_concept_data):
        """Test listing algorithms."""
        concept_repo = test_db["concepts"]
        algo_repo = test_db["algorithms"]

        # Create concept
        concept = concept_repo.create_concept(**sample_concept_data)

        # Create multiple algorithms
        for i in range(3):
            data = sample_algorithm_data.copy()
            data["name"] = f"{data['name']}_{i}"
            data["display_name"] = f"{data['display_name']} {i}"
            algo_repo.create_algorithm(
                concept_id=concept.id,
                **data,
            )

        algorithms = algo_repo.list_algorithms(concept_id=concept.id, skip=0, limit=10)

        assert len(algorithms) == 3
        assert all(algo.name.startswith("test_algorithm") for algo in algorithms)


class TestSavedModelRepository:
    """Test SavedModelRepository operations."""

    def test_save_model(self, test_db, sample_user_data):
        """Test saving a model."""
        user_repo = test_db["users"]
        model_repo = test_db["models"]

        # Create user
        user = user_repo.create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
        )

        # Save model
        model_data = {
            "coefficients": [1.0, 2.0, 3.0],
            "intercept": 0.5,
        }

        model = model_repo.save_model(
            user_id=user.id,
            name="Test Model",
            model_type="regression",
            algorithm="linear_regression",
            model_data=model_data,
            metrics={"mse": 0.1, "r2": 0.9},
            feature_names=["feature1", "feature2", "feature3"],
            target_name="target",
            is_public=True,
        )

        assert model.id is not None
        assert model.name == "Test Model"
        assert model.model_type == "regression"
        assert model.algorithm == "linear_regression"
        assert model.model_data == model_data
        assert model.metrics == {"mse": 0.1, "r2": 0.9}
        assert model.feature_names == ["feature1", "feature2", "feature3"]
        assert model.target_name == "target"
        assert model.is_public is True

    def test_get_model(self, test_db, sample_user_data):
        """Test getting a saved model by ID."""
        user_repo = test_db["users"]
        model_repo = test_db["models"]

        # Create user and model
        user = user_repo.create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
        )

        model = model_repo.save_model(
            user_id=user.id,
            name="Test Model",
            model_type="regression",
            algorithm="linear_regression",
            model_data={},
            metrics={},
        )

        retrieved_model = model_repo.get_model(model.id)

        assert retrieved_model is not None
        assert retrieved_model.id == model.id
        assert retrieved_model.name == model.name

    def test_list_models(self, test_db, sample_user_data):
        """Test listing saved models."""
        user_repo = test_db["users"]
        model_repo = test_db["models"]

        # Create user
        user = user_repo.create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
        )

        # Create multiple models
        for i in range(3):
            model_repo.save_model(
                user_id=user.id,
                name=f"Test Model {i}",
                model_type="regression",
                algorithm="linear_regression",
                model_data={},
                metrics={},
            )

        models = model_repo.list_models(user_id=user.id, skip=0, limit=10)

        assert len(models) == 3
        assert all(model.name.startswith("Test Model") for model in models)

    def test_list_models_by_type(self, test_db, sample_user_data):
        """Test listing saved models by type."""
        user_repo = test_db["users"]
        model_repo = test_db["models"]

        # Create user
        user = user_repo.create_user(
            username=sample_user_data["username"],
            email=sample_user_data["email"],
        )

        # Create models of different types
        model_repo.save_model(
            user_id=user.id,
            name="Regression Model",
            model_type="regression",
            algorithm="linear_regression",
            model_data={},
            metrics={},
        )

        model_repo.save_model(
            user_id=user.id,
            name="Classification Model",
            model_type="classification",
            algorithm="random_forest",
            model_data={},
            metrics={},
        )

        # List only regression models
        regression_models = model_repo.list_models(
            user_id=user.id,
            model_type="regression",
            skip=0,
            limit=10,
        )

        assert len(regression_models) == 1
        assert regression_models[0].name == "Regression Model"
        assert regression_models[0].model_type == "regression"


class TestDatabaseIntegration:
    """Integration tests for database operations."""

    def test_user_experiment_relationship(
        self, test_db, sample_user_data, sample_experiment_data
    ):
        """Test user-experiment relationship."""
        user_repo = test_db["users"]
        exp_repo = test_db["experiments"]

        # Create user
        user = user_repo.create_user(**sample_user_data)

        # Create experiment for user
        experiment = exp_repo.create_experiment(
            user_id=user.id,
            **sample_experiment_data,
        )

        # Verify relationship
        assert experiment.user_id == user.id

        # User should have experiments
        # Note: This requires loading the relationship
        # In practice, you'd query experiments by user_id

    def test_concept_algorithm_relationship(
        self, test_db, sample_concept_data, sample_algorithm_data
    ):
        """Test concept-algorithm relationship."""
        concept_repo = test_db["concepts"]
        algo_repo = test_db["algorithms"]

        # Create concept
        concept = concept_repo.create_concept(**sample_concept_data)

        # Create algorithm for concept
        algorithm = algo_repo.create_algorithm(
            concept_id=concept.id,
            **sample_algorithm_data,
        )

        # Verify relationship
        assert algorithm.concept_id == concept.id

        # List algorithms for concept
        algorithms = algo_repo.list_algorithms(concept_id=concept.id)
        assert len(algorithms) == 1
        assert algorithms[0].id == algorithm.id

    def test_experiment_without_user(self, test_db, sample_experiment_data):
        """Test creating experiment without user (anonymous)."""
        exp_repo = test_db["experiments"]

        # Create experiment without user_id
        experiment = exp_repo.create_experiment(
            user_id=None,
            **sample_experiment_data,
        )

        assert experiment.id is not None
        assert experiment.user_id is None
        assert experiment.name == sample_experiment_data["name"]

    def test_duplicate_username(self, test_db, sample_user_data):
        """Test duplicate username constraint."""
        user_repo = test_db["users"]

        # Create first user
        user_repo.create_user(**sample_user_data)

        # Try to create user with same username
        # This should raise an integrity error
        import sqlalchemy.exc

        with pytest.raises(sqlalchemy.exc.IntegrityError):
            user_repo.create_user(**sample_user_data)

    def test_public_vs_private(self, test_db, sample_experiment_data, sample_user_data):
        """Test public vs private experiments."""
        user_repo = test_db["users"]
        exp_repo = test_db["experiments"]

        # Create user
        user = user_repo.create_user(**sample_user_data)

        # Create public experiment
        public_data = sample_experiment_data.copy()
        public_data["is_public"] = True
        public_exp = exp_repo.create_experiment(
            user_id=user.id,
            **public_data,
        )

        # Create private experiment
        private_data = sample_experiment_data.copy()
        private_data["name"] = "Private Experiment"
        private_data["is_public"] = False
        private_exp = exp_repo.create_experiment(
            user_id=user.id,
            **private_data,
        )

        # List public experiments only
        public_experiments = exp_repo.list_experiments(
            user_id=user.id,
            public_only=True,
        )

        assert len(public_experiments) == 1
        assert public_experiments[0].id == public_exp.id
        assert public_experiments[0].is_public is True

        # List all experiments (public and private)
        all_experiments = exp_repo.list_experiments(
            user_id=user.id,
            public_only=False,
        )

        assert len(all_experiments) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
