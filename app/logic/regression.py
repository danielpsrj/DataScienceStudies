"""
Linear regression logic and utilities.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.graph_objects as go
import plotly.express as px

import streamlit as st
from app.caching import EnhancedCache


def generate_linear_data(
    n_samples: int = 100,
    n_features: int = 1,
    noise: float = 0.1,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic linear regression data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Standard deviation of Gaussian noise
        random_state: Random seed
        
    Returns:
        Tuple of (X, y) arrays
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate coefficients
    true_coef = np.random.randn(n_features)
    
    # Generate target with noise
    y = X @ true_coef + np.random.normal(0, noise, n_samples)
    
    return X, y


def train_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train a linear regression model and return results.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Dictionary with model results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get coefficients
    if X.shape[1] == 1:
        coefficients = {"slope": model.coef_[0]}
    else:
        coefficients = {f"coef_{i}": coef for i, coef in enumerate(model.coef_)}
    
    return {
        "model": model,
        "intercept": model.intercept_,
        "coefficients": coefficients,
        "mse": mse,
        "r2": r2,
        "y_test": y_test,
        "y_pred": y_pred,
        "X_test": X_test,
    }


def train_regularized_regression(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1.0,
    regularization: str = "ridge",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    """
    Train regularized regression (Ridge or Lasso).
    
    Args:
        X: Feature matrix
        y: Target vector
        alpha: Regularization strength
        regularization: 'ridge' or 'lasso'
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Dictionary with model results
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Train model
    if regularization == "ridge":
        model = Ridge(alpha=alpha)
    elif regularization == "lasso":
        model = Lasso(alpha=alpha)
    else:
        raise ValueError(f"Unknown regularization: {regularization}")
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Get coefficients
    if X.shape[1] == 1:
        coefficients = {"slope": model.coef_[0]}
    else:
        coefficients = {f"coef_{i}": coef for i, coef in enumerate(model.coef_)}
    
    return {
        "model": model,
        "intercept": model.intercept_,
        "coefficients": coefficients,
        "mse": mse,
        "r2": r2,
        "alpha": alpha,
        "regularization": regularization,
        "y_test": y_test,
        "y_pred": y_pred,
        "X_test": X_test,
    }


def plot_regression_results(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    show_residuals: bool = True,
) -> go.Figure:
    """
    Create regression plot with optional residuals.
    
    Args:
        X: Feature matrix (1D or 2D)
        y: True target values
        y_pred: Predicted target values
        show_residuals: Whether to show residual plot
        
    Returns:
        Plotly figure
    """
    if X.shape[1] == 1:
        # Simple linear regression plot
        X_flat = X.flatten()
        
        if show_residuals:
            # Create subplots
            from plotly.subplots import make_subplots
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Regression Line", "Residuals"),
                vertical_spacing=0.15
            )
            
            # Regression line
            fig.add_trace(
                go.Scatter(
                    x=X_flat, y=y,
                    mode="markers",
                    name="Data",
                    marker=dict(color="blue", size=8, opacity=0.6)
                ),
                row=1, col=1
            )
            
            # Sort for line plot
            sort_idx = np.argsort(X_flat)
            fig.add_trace(
                go.Scatter(
                    x=X_flat[sort_idx], y=y_pred[sort_idx],
                    mode="lines",
                    name="Regression Line",
                    line=dict(color="red", width=3)
                ),
                row=1, col=1
            )
            
            # Residuals
            residuals = y - y_pred
            fig.add_trace(
                go.Scatter(
                    x=X_flat, y=residuals,
                    mode="markers",
                    name="Residuals",
                    marker=dict(color="green", size=8, opacity=0.6)
                ),
                row=2, col=1
            )
            
            # Zero line for residuals
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
            
            fig.update_layout(height=600, showlegend=True)
            
        else:
            # Single plot
            fig = go.Figure()
            
            fig.add_trace(
                go.Scatter(
                    x=X_flat, y=y,
                    mode="markers",
                    name="Data",
                    marker=dict(color="blue", size=8, opacity=0.6)
                )
            )
            
            # Sort for line plot
            sort_idx = np.argsort(X_flat)
            fig.add_trace(
                go.Scatter(
                    x=X_flat[sort_idx], y=y_pred[sort_idx],
                    mode="lines",
                    name="Regression Line",
                    line=dict(color="red", width=3)
                )
            )
            
            fig.update_layout(
                title="Linear Regression",
                xaxis_title="Feature",
                yaxis_title="Target",
                height=500,
            )
    
    else:
        # Multiple regression - show predicted vs actual
        fig = go.Figure()
        
        fig.add_trace(
            go.Scatter(
                x=y, y=y_pred,
                mode="markers",
                name="Predictions",
                marker=dict(color="blue", size=8, opacity=0.6)
            )
        )
        
        # Perfect prediction line
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode="lines",
                name="Perfect Prediction",
                line=dict(color="red", width=2, dash="dash")
            )
        )
        
        fig.update_layout(
            title="Predicted vs Actual",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=500,
        )
    
    return fig


def compare_regularization(
    X: np.ndarray,
    y: np.ndarray,
    alphas: list[float] = [0.01, 0.1, 1.0, 10.0],
    test_size: float = 0.2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compare different regularization strengths.
    
    Args:
        X: Feature matrix
        y: Target vector
        alphas: List of regularization strengths
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for alpha in alphas:
        # Ridge
        ridge_results = train_regularized_regression(
            X, y, alpha=alpha, regularization="ridge",
            test_size=test_size, random_state=random_state
        )
        
        # Lasso
        lasso_results = train_regularized_regression(
            X, y, alpha=alpha, regularization="lasso",
            test_size=test_size, random_state=random_state
        )
        
        results.append({
            "alpha": alpha,
            "ridge_mse": ridge_results["mse"],
            "ridge_r2": ridge_results["r2"],
            "lasso_mse": lasso_results["mse"],
            "lasso_r2": lasso_results["r2"],
            "ridge_coef_norm": np.linalg.norm(ridge_results["model"].coef_),
            "lasso_coef_norm": np.linalg.norm(lasso_results["model"].coef_),
        })
    
    return pd.DataFrame(results)


# Cached versions for performance
@st.cache_data(ttl=300)
def cached_generate_linear_data(n_samples: int = 100, noise: float = 0.1) -> tuple:
    """Cached version of generate_linear_data."""
    return generate_linear_data(n_samples=n_samples, noise=noise)


@st.cache_data(ttl=300)
def cached_train_linear_regression(X: np.ndarray, y: np.ndarray) -> dict:
    """Cached version of train_linear_regression."""
    return train_linear_regression(X, y)


# Example usage:
"""
# Generate data
X, y = generate_linear_data(n_samples=100, noise=0.2)

# Train model
results = train_linear_regression(X, y)
print(f"R² score: {results['r2']:.3f}")
print(f"MSE: {results['mse']:.3f}")

# Create plot
fig = plot_regression_results(
    results['X_test'], 
    results['y_test'], 
    results['y_pred'],
    show_residuals=True
)
fig.show()

# Compare regularization
df_comparison = compare_regularization(X, y)
print(df_comparison)
"""