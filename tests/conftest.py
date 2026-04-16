"""
Pytest configuration and fixtures for the Data Science Platform tests.
"""

import pytest
import sys
import os
from pathlib import Path
from typing import Generator, Any, Dict
import tempfile
import shutil

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import Settings
from app.data.repositories import db as main_db
from app.data.models import Base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Test settings with in-memory SQLite database."""
    return Settings(
        app_name="Data Science Platform Test",
        app_env="test",
        debug=True,
        database_url="sqlite:///:memory:",
        api_host="127.0.0.1",
        api_port=9999,
        streamlit_host="127.0.0.1",
        streamlit_port=9998,
    )


@pytest.fixture(scope="session")
def test_engine(test_settings: Settings):
    """Create a test database engine."""
    engine = create_engine(test_settings.database_url, echo=False)

    # Create all tables
    Base.metadata.create_all(bind=engine)

    yield engine

    # Cleanup
    engine.dispose()


@pytest.fixture
def test_session(test_engine) -> Generator:
    """Create a test database session."""
    Session = sessionmaker(bind=test_engine)
    session = Session()

    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def test_db(test_session):
    """Test database with repositories."""
    from app.data.repositories import (
        UserRepository,
        ExperimentRepository,
        SavedModelRepository,
        DatasetRepository,
        ConceptRepository,
        AlgorithmRepository,
    )

    # Create test repositories with test session
    return {
        "session": test_session,
        "users": UserRepository(session=test_session),
        "experiments": ExperimentRepository(session=test_session),
        "models": SavedModelRepository(session=test_session),
        "datasets": DatasetRepository(session=test_session),
        "concepts": ConceptRepository(session=test_session),
        "algorithms": AlgorithmRepository(session=test_session),
    }


@pytest.fixture
def sample_regression_data():
    """Generate sample regression data for testing."""
    import numpy as np

    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = 2 * X[:, 0] + 1.5 * X[:, 1] + np.random.normal(0, 0.1, 100)

    return {"X": X, "y": y}


@pytest.fixture
def sample_clustering_data():
    """Generate sample clustering data for testing."""
    import numpy as np

    np.random.seed(42)
    # Create 3 clusters
    cluster1 = np.random.randn(50, 2) + np.array([0, 0])
    cluster2 = np.random.randn(50, 2) + np.array([5, 5])
    cluster3 = np.random.randn(50, 2) + np.array([-5, 5])

    X = np.vstack([cluster1, cluster2, cluster3])
    y_true = np.hstack([np.zeros(50), np.ones(50), np.full(50, 2)])

    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y_true = y_true[shuffle_idx]

    return {"X": X, "y_true": y_true}


@pytest.fixture
def sample_user_data() -> Dict[str, Any]:
    """Sample user data for testing."""
    return {
        "username": "test_user",
        "email": "test@example.com",
    }


@pytest.fixture
def sample_experiment_data() -> Dict[str, Any]:
    """Sample experiment data for testing."""
    return {
        "name": "Test Experiment",
        "description": "Test experiment description",
        "concept_type": "regression",
        "algorithm": "linear_regression",
        "parameters": {
            "n_samples": 100,
            "n_features": 2,
            "test_size": 0.2,
            "random_state": 42,
        },
        "metrics": {
            "mse": 0.1,
            "r2": 0.85,
            "mae": 0.25,
        },
        "dataset_info": {
            "name": "test_dataset",
            "n_rows": 100,
            "n_features": 2,
            "target": "target",
        },
        "is_public": True,
    }


@pytest.fixture
def sample_dataset_data() -> Dict[str, Any]:
    """Sample dataset data for testing."""
    return {
        "name": "test_dataset",
        "description": "Test dataset description",
        "dataset_type": "synthetic",
        "source": "generated",
        "columns": ["feature_1", "feature_2", "target"],
        "num_rows": 100,
        "num_features": 2,
        "target_column": "target",
        "is_public": True,
    }


@pytest.fixture
def sample_concept_data() -> Dict[str, Any]:
    """Sample concept data for testing."""
    return {
        "name": "test_concept",
        "display_name": "Test Concept",
        "category": "supervised",
        "description": "Test concept description",
        "difficulty_level": "beginner",
        "mathematical_formulation": "y = f(X) + ε",
        "common_algorithms": ["algorithm1", "algorithm2"],
        "use_cases": ["use_case1", "use_case2"],
        "prerequisites": ["prereq1", "prereq2"],
    }


@pytest.fixture
def sample_algorithm_data() -> Dict[str, Any]:
    """Sample algorithm data for testing."""
    return {
        "name": "test_algorithm",
        "display_name": "Test Algorithm",
        "description": "Test algorithm description",
        "parameters": {
            "param1": {"type": "float", "default": 1.0, "description": "Parameter 1"},
            "param2": {"type": "int", "default": 10, "description": "Parameter 2"},
        },
        "strengths": ["strength1", "strength2"],
        "weaknesses": ["weakness1", "weakness2"],
        "implementation_libraries": ["library1", "library2"],
    }


@pytest.fixture
def temp_dir() -> Generator:
    """Create a temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def cleanup_test_database(test_engine):
    """Clean up database after each test."""
    yield
    # Drop all tables and recreate
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)


# Markers for test categorization
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: mark test as slow (skip with -m 'not slow')"
    )
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "unit: mark test as unit test")
    config.addinivalue_line("markers", "database: mark test as database test")
    config.addinivalue_line("markers", "api: mark test as API test")
    config.addinivalue_line("markers", "logic: mark test as logic test")
    config.addinivalue_line("markers", "component: mark test as component test")


# Skip tests that require external services in CI
def pytest_collection_modifyitems(config, items):
    """Skip tests based on environment."""
    skip_slow = pytest.mark.skip(reason="slow test - run with -m 'not slow'")
    skip_external = pytest.mark.skip(reason="requires external services")

    for item in items:
        # Check if --run-slow option exists before using it
        try:
            run_slow = config.getoption("--run-slow")
            if "slow" in item.keywords and run_slow is False:
                item.add_marker(skip_slow)
        except ValueError:
            # Option doesn't exist, skip slow tests by default
            if "slow" in item.keywords:
                item.add_marker(skip_slow)

        # Skip tests that require internet if OFFLINE mode
        if "requires_internet" in item.keywords and os.getenv("OFFLINE", "0") == "1":
            item.add_marker(skip_external)
