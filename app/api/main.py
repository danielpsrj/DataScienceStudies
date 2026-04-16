"""
FastAPI service for the Data Science Platform.
Provides REST API endpoints for data science concepts and operations.
"""
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
import json

from app.config import settings
from app.logic.regression import (
    generate_linear_data,
    train_linear_regression,
    compare_regularization,
)

# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="API for Data Science Platform concepts and operations",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_hosts_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information."""
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "description": "Data Science Platform API",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "concepts": "/concepts",
            "regression": "/regression/*",
        },
        "environment": settings.app_env,
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "version": "0.1.0",
        "environment": settings.app_env,
    }


@app.get("/concepts")
async def list_concepts() -> Dict[str, List[str]]:
    """List available data science concepts."""
    return {
        "concepts": [
            "linear_regression",
            "logistic_regression",
            "kmeans_clustering",
            "decision_trees",
            "random_forest",
            "svm",
            "neural_networks",
        ],
        "available_endpoints": [
            "/regression/linear",
            "/regression/logistic",
            "/clustering/kmeans",
        ],
    }


@app.get("/concepts/{concept_name}")
async def get_concept_info(concept_name: str) -> Dict[str, Any]:
    """Get information about a specific concept."""
    concepts = {
        "linear_regression": {
            "name": "Linear Regression",
            "description": "Supervised learning algorithm for predicting continuous values",
            "type": "regression",
            "category": "supervised_learning",
            "mathematical_formulation": "y = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ + ε",
            "common_use_cases": [
                "House price prediction",
                "Sales forecasting",
                "Risk assessment",
            ],
            "parameters": ["sample_size", "noise_level", "test_size"],
        },
        "kmeans_clustering": {
            "name": "K-Means Clustering",
            "description": "Unsupervised learning algorithm for grouping similar data points",
            "type": "clustering",
            "category": "unsupervised_learning",
            "mathematical_formulation": "argmin_S Σᵢ=1ᵏ Σ_{x∈Sᵢ} ||x - μᵢ||²",
            "common_use_cases": [
                "Customer segmentation",
                "Image compression",
                "Anomaly detection",
            ],
            "parameters": ["n_clusters", "max_iter", "random_state"],
        },
    }
    
    if concept_name not in concepts:
        raise HTTPException(status_code=404, detail=f"Concept '{concept_name}' not found")
    
    return concepts[concept_name]


# Regression endpoints
@app.get("/regression/linear/generate")
async def generate_linear_regression_data(
    n_samples: int = Query(100, ge=10, le=10000, description="Number of samples"),
    n_features: int = Query(1, ge=1, le=10, description="Number of features"),
    noise: float = Query(0.1, ge=0.0, le=1.0, description="Noise level"),
    random_state: Optional[int] = Query(None, description="Random seed"),
) -> Dict[str, Any]:
    """Generate synthetic linear regression data."""
    try:
        X, y = generate_linear_data(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state or 42,
        )
        
        # Convert to lists for JSON serialization
        if n_features == 1:
            X_list = X.flatten().tolist()
        else:
            X_list = X.tolist()
        
        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "noise": noise,
            "random_state": random_state or 42,
            "X": X_list,
            "y": y.tolist(),
            "summary": {
                "X_mean": float(X.mean()),
                "X_std": float(X.std()),
                "y_mean": float(y.mean()),
                "y_std": float(y.std()),
                "correlation": float(np.corrcoef(X.flatten(), y)[0, 1]) if n_features == 1 else None,
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating data: {str(e)}")


@app.post("/regression/linear/train")
async def train_linear_regression_model(
    X: List[List[float]],
    y: List[float],
    test_size: float = Query(0.2, ge=0.1, le=0.5, description="Test set proportion"),
    random_state: Optional[int] = Query(None, description="Random seed"),
) -> Dict[str, Any]:
    """Train a linear regression model."""
    try:
        # Convert to numpy arrays
        X_array = np.array(X)
        y_array = np.array(y)
        
        # Validate input shapes
        if len(X_array.shape) == 1:
            X_array = X_array.reshape(-1, 1)
        
        if X_array.shape[0] != len(y_array):
            raise ValueError(f"X has {X_array.shape[0]} samples but y has {len(y_array)}")
        
        # Train model
        results = train_linear_regression(
            X_array, y_array,
            test_size=test_size,
            random_state=random_state or 42,
        )
        
        # Prepare response
        response = {
            "model_type": "linear_regression",
            "test_size": test_size,
            "random_state": random_state or 42,
            "metrics": {
                "mse": float(results["mse"]),
                "r2": float(results["r2"]),
                "intercept": float(results["intercept"]),
            },
            "coefficients": {k: float(v) for k, v in results["coefficients"].items()},
            "predictions": {
                "y_test": results["y_test"].tolist(),
                "y_pred": results["y_pred"].tolist(),
            },
        }
        
        # Add feature names if available
        if X_array.shape[1] > 1:
            response["feature_importance"] = {
                f"feature_{i}": float(abs(coef))
                for i, coef in enumerate(results["model"].coef_)
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error training model: {str(e)}")


@app.get("/regression/linear/compare-regularization")
async def compare_regularization_methods(
    n_samples: int = Query(100, ge=10, le=1000, description="Number of samples"),
    n_features: int = Query(5, ge=2, le=20, description="Number of features"),
    noise: float = Query(0.2, ge=0.0, le=1.0, description="Noise level"),
    alphas: str = Query("0.01,0.1,1.0,10.0", description="Comma-separated alpha values"),
    test_size: float = Query(0.2, ge=0.1, le=0.5, description="Test set proportion"),
    random_state: Optional[int] = Query(None, description="Random seed"),
) -> Dict[str, Any]:
    """Compare Ridge and Lasso regularization."""
    try:
        # Parse alphas
        alpha_list = [float(a.strip()) for a in alphas.split(",")]
        
        # Generate data
        X, y = generate_linear_data(
            n_samples=n_samples,
            n_features=n_features,
            noise=noise,
            random_state=random_state or 42,
        )
        
        # Compare regularization
        df_comparison = compare_regularization(
            X, y,
            alphas=alpha_list,
            test_size=test_size,
            random_state=random_state or 42,
        )
        
        # Convert to dictionary
        results = df_comparison.to_dict(orient="records")
        
        return {
            "n_samples": n_samples,
            "n_features": n_features,
            "noise": noise,
            "alphas": alpha_list,
            "test_size": test_size,
            "comparison": results,
            "best_ridge": {
                "alpha": float(df_comparison.loc[df_comparison["ridge_r2"].idxmax(), "alpha"]),
                "r2": float(df_comparison["ridge_r2"].max()),
            },
            "best_lasso": {
                "alpha": float(df_comparison.loc[df_comparison["lasso_r2"].idxmax(), "alpha"]),
                "r2": float(df_comparison["lasso_r2"].max()),
            },
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error comparing regularization: {str(e)}")


# Example endpoint
@app.get("/examples/regression")
async def regression_example() -> Dict[str, Any]:
    """Example regression workflow."""
    # Generate data
    X, y = generate_linear_data(n_samples=100, noise=0.2)
    
    # Train model
    results = train_linear_regression(X, y)
    
    return {
        "example": "linear_regression_workflow",
        "data_generation": {
            "n_samples": 100,
            "noise": 0.2,
            "data_summary": {
                "X_shape": X.shape,
                "y_shape": y.shape,
                "X_mean": float(X.mean()),
                "y_mean": float(y.mean()),
            },
        },
        "model_training": {
            "test_size": 0.2,
            "metrics": {
                "mse": float(results["mse"]),
                "r2": float(results["r2"]),
            },
            "coefficients": {k: float(v) for k, v in results["coefficients"].items()},
        },
        "code_example": {
            "python": '''import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate data
X = np.random.randn(100, 1)
y = 2 * X.squeeze() + np.random.normal(0, 0.2, 100)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = np.mean((y_test - y_pred) ** 2)
r2 = 1 - mse / np.var(y_test)

print(f"Slope: {model.coef_[0]:.3f}")
print(f"Intercept: {model.intercept_:.3f}")
print(f"MSE: {mse:.3f}, R²: {r2:.3f}")''',
        },
    }


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": request.url.path,
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": str(exc),
            "type": type(exc).__name__,
            "status_code": 500,
            "path": request.url.path,
        },
    )


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=settings.api_workers if not settings.debug else 1,
    )