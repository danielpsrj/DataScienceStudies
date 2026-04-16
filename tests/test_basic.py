"""
Basic tests for the Data Science Platform.
"""
import pytest
import numpy as np
import pandas as pd

from app.logic.regression import (
    generate_linear_data,
    train_linear_regression,
    plot_regression_results,
)


def test_generate_linear_data():
    """Test linear data generation."""
    X, y = generate_linear_data(n_samples=100, noise=0.1, random_state=42)
    
    assert X.shape == (100, 1)
    assert y.shape == (100,)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    
    # Test with multiple features
    X_multi, y_multi = generate_linear_data(n_samples=50, n_features=3, random_state=42)
    assert X_multi.shape == (50, 3)
    assert y_multi.shape == (50,)


def test_train_linear_regression():
    """Test linear regression training."""
    X, y = generate_linear_data(n_samples=100, noise=0.1, random_state=42)
    results = train_linear_regression(X, y, test_size=0.2, random_state=42)
    
    assert "model" in results
    assert "mse" in results
    assert "r2" in results
    assert "intercept" in results
    assert "coefficients" in results
    
    # Check types
    assert isinstance(results["mse"], float)
    assert isinstance(results["r2"], float)
    assert isinstance(results["intercept"], float)
    assert isinstance(results["coefficients"], dict)
    
    # Check reasonable values
    assert results["r2"] >= -1.0 and results["r2"] <= 1.0
    assert results["mse"] >= 0.0


def test_plot_regression_results():
    """Test regression plot generation."""
    X, y = generate_linear_data(n_samples=50, noise=0.1, random_state=42)
    results = train_linear_regression(X, y, test_size=0.2, random_state=42)
    
    # Test with residuals
    fig_with_residuals = plot_regression_results(
        results["X_test"],
        results["y_test"],
        results["y_pred"],
        show_residuals=True
    )
    
    # Test without residuals
    fig_without_residuals = plot_regression_results(
        results["X_test"],
        results["y_test"],
        results["y_pred"],
        show_residuals=False
    )
    
    # Check that figures are created
    assert fig_with_residuals is not None
    assert fig_without_residuals is not None
    
    # Check figure properties
    assert hasattr(fig_with_residuals, "data")
    assert hasattr(fig_without_residuals, "data")


def test_config_import():
    """Test configuration import."""
    from app.config import settings
    
    assert hasattr(settings, "app_name")
    assert hasattr(settings, "app_env")
    assert hasattr(settings, "debug")
    assert hasattr(settings, "database_url")
    
    # Check default values
    assert settings.app_name == "Data Science Platform"
    assert settings.app_env == "development"
    assert settings.debug is True


def test_state_management():
    """Test state management import."""
    from app.state import get_state, AppState
    
    # Test class exists
    assert AppState is not None
    
    # Test function exists
    assert callable(get_state)


def test_caching_import():
    """Test caching import."""
    from app.caching import (
        EnhancedCache,
        cache_data,
        get_cached_data,
        clear_all_caches,
    )
    
    # Test classes and functions exist
    assert EnhancedCache is not None
    assert callable(cache_data)
    assert callable(get_cached_data)
    assert callable(clear_all_caches)


@pytest.mark.slow
def test_regression_performance():
    """Performance test for regression (marked as slow)."""
    import time
    
    # Generate larger dataset
    X, y = generate_linear_data(n_samples=1000, noise=0.2, random_state=42)
    
    start_time = time.time()
    results = train_linear_regression(X, y, test_size=0.2, random_state=42)
    end_time = time.time()
    
    execution_time = end_time - start_time
    
    # Should complete within 2 seconds
    assert execution_time < 2.0, f"Regression took {execution_time:.2f} seconds"
    
    # Check results are reasonable
    assert results["r2"] > 0.5  # With low noise, R² should be decent


if __name__ == "__main__":
    # Run tests directly
    test_generate_linear_data()
    print("✓ test_generate_linear_data passed")
    
    test_train_linear_regression()
    print("✓ test_train_linear_regression passed")
    
    test_config_import()
    print("✓ test_config_import passed")
    
    print("All basic tests passed!")