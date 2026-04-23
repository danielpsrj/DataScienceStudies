"""
Regression-specific demo component.
Contains demo logic specific to the Linear Regression page.
"""

import streamlit as st
from app.logic.regression import (
    generate_linear_data,
    plot_regression_results,
    train_linear_regression,
)


def regression_demo() -> None:
    """Interactive demo for linear regression."""
    with st.expander("🎮 Interactive Demo", expanded=True):
        # Demo controls at the top (horizontal layout)
        
        # Create horizontal columns for controls
        control_col1, control_col2, control_col3, control_col4 = st.columns(4)
        
        with control_col1:
            sample_size = st.slider(
                "Sample Size",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Number of data points to generate",
            )
        
        with control_col2:
            noise_level = st.slider(
                "Noise Level",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                help="Standard deviation of Gaussian noise",
            )
        
        with control_col3:
            test_size = st.slider(
                "Test Size",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
                help="Proportion of data for testing",
            )
        
        with control_col4:
            show_residuals = st.checkbox(
                "Show Residuals",
                value=True,
                help="Display residual plot below regression line",
            )
        
        # Run button centered below controls
        run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
        with run_col2:
            if st.button("🚀 Run Linear Regression", type="primary", use_container_width=True):
                with st.spinner("Generating data and training model..."):
                    # Generate data
                    X, y = generate_linear_data(
                        n_samples=sample_size,
                        n_features=1,
                        noise=noise_level,
                        random_state=42,
                    )
                    
                    # Train model
                    results = train_linear_regression(
                        X, y, test_size=test_size, random_state=42
                    )
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("R² Score", f"{results['r2']:.4f}")
                        st.metric("Mean Squared Error", f"{results['mse']:.4f}")
                    
                    with col2:
                        st.metric("Slope (β₁)", f"{results['coefficients']['slope']:.4f}")
                        st.metric("Intercept (β₀)", f"{results['intercept']:.4f}")
                    
                    # Plot results
                    fig = plot_regression_results(
                        results["X_test"],
                        results["y_test"],
                        results["y_pred"],
                        show_residuals=show_residuals,
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show raw data
                    with st.expander("📊 Generated Data", expanded=False):
                        import pandas as pd
                        
                        df = pd.DataFrame({"Feature": X.flatten(), "Target": y})
                        st.dataframe(df.head(10))
                        st.caption(f"Showing 10 of {len(df)} rows")


def get_regression_demo_code() -> str:
    """Get code snippet for the regression demo."""
    return '''
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def run_regression_demo(sample_size=100, noise_level=0.2, test_size=0.2):
    # Generate synthetic data
    X = np.random.randn(sample_size, 1)
    true_slope = 2.0
    true_intercept = 1.0
    y = true_intercept + true_slope * X.squeeze() + np.random.normal(0, noise_level, sample_size)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        "coefficients": {"slope": model.coef_[0]},
        "intercept": model.intercept_,
        "mse": mse,
        "r2": r2,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }
'''