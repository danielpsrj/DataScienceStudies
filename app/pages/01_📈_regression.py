"""
Linear Regression concept page.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from app.components import (
    theory_section,
    math_equation,
    code_tabs,
    display_applications,
    display_pitfalls,
    display_references,
    get_regression_references,
)
from app.logic.regression import (
    generate_linear_data,
    train_linear_regression,
    plot_regression_results,
    compare_regularization,
)
from app.state import get_state


def main() -> None:
    """Main page function."""
    # Page header
    st.title("📈 Linear Regression")
    st.caption("A supervised learning algorithm for predicting continuous values")
    
    # Track page visit
    state = get_state()
    state.add_to_history("linear_regression")
    state.current_model = "linear_regression"
    
    # 1. Concept Overview
    theory_section(
        title="Concept Overview",
        content="""
        **Linear Regression** is a fundamental supervised learning algorithm that models 
        the relationship between a dependent variable (target) and one or more independent 
        variables (features) using a linear approach.
        
        The goal is to find the best-fitting straight line (or hyperplane in higher dimensions) 
        through the data points by minimizing the sum of squared residuals (differences between 
        observed and predicted values).
        
        ### Key Characteristics:
        - **Simple yet powerful** for modeling linear relationships
        - **Interpretable** coefficients show feature importance
        - **Fast to train** with closed-form solution
        - **Foundation** for more complex models
        
        Linear regression serves as a building block for many advanced techniques and 
        provides valuable insights into the relationship between variables.
        """,
        image_url="https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Linear_regression.svg/800px-Linear_regression.svg.png",
        columns=(2, 1),
    )
    
    # Mathematical formulation (not in expander since content is short)
    math_equation(
        equation=r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_n x_n + \epsilon",
        variables={
            "y": "Dependent variable (target)",
            "\\beta_0": "Intercept (bias term)",
            "\\beta_i": "Coefficients (feature weights)",
            "x_i": "Independent variables (features)",
            "\\epsilon": "Error term (residuals)"
        },
        title="Mathematical Formulation",
        icon="🧮",
        expandable=False,
    )
    
    # Matrix formulation (not in expander since content is short)
    st.markdown("**Matrix Formulation:**")
    st.latex(r"\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}")
    st.markdown("**Where:**")
    st.markdown(r"- $\mathbf{y}$: Vector of target values $(n \times 1)$")
    st.markdown(r"- $\mathbf{X}$: Design matrix $(n \times p)$")
    st.markdown(r"- $\boldsymbol{\beta}$: Coefficient vector $(p \times 1)$")
    st.markdown(r"- $\boldsymbol{\epsilon}$: Error vector $(n \times 1)$")
    
    st.markdown("**Ordinary Least Squares (OLS) Solution:**")
    st.latex(r"\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}")
    
    # 2. Interactive Demo
    st.header("🎮 Interactive Demo")
    
    # Create columns for demo layout
    demo_col1, demo_col2 = st.columns([1, 2])
    
    with demo_col1:
        st.subheader("⚙️ Regression Parameters")
        
        sample_size = st.slider(
            "Sample Size",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Number of data points to generate"
        )
        
        noise_level = st.slider(
            "Noise Level",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
            help="Standard deviation of Gaussian noise"
        )
        
        test_size = st.slider(
            "Test Size",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Proportion of data for testing"
        )
        
        show_residuals = st.checkbox(
            "Show Residuals",
            value=True,
            help="Display residual plot below regression line"
        )
    
    with demo_col2:
        # Run demo button
        if st.button("🚀 Run Linear Regression", type="primary", use_container_width=True):
            with st.spinner("Generating data and training model..."):
                # Generate data
                X, y = generate_linear_data(
                    n_samples=sample_size,
                    n_features=1,
                    noise=noise_level,
                    random_state=42
                )
                
                # Train model
                results = train_linear_regression(
                    X, y,
                    test_size=test_size,
                    random_state=42
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
                    results['X_test'],
                    results['y_test'],
                    results['y_pred'],
                    show_residuals=show_residuals
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show raw data
                with st.expander("📊 Generated Data", expanded=False):
                    import pandas as pd
                    df = pd.DataFrame({
                        'Feature': X.flatten(),
                        'Target': y
                    })
                    st.dataframe(df.head(10))
                    st.caption(f"Showing 10 of {len(df)} rows")
    
    # 3. Implementation Examples
    st.header("💻 Implementation Examples")
    
    code_tabs({
        "Scikit-learn": """
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
print(f"MSE: {mse:.4f}, R²: {r2:.4f}")
""",
        "NumPy (OLS)": """
import numpy as np

# Add intercept column
X_with_intercept = np.c_[np.ones(X.shape[0]), X]

# Normal equation: β = (XᵀX)⁻¹Xᵀy
beta = np.linalg.inv(X_with_intercept.T @ X_with_intercept) @ X_with_intercept.T @ y

# Extract coefficients
intercept = beta[0]
coefficients = beta[1:]

print(f"Intercept: {intercept:.4f}")
print(f"Coefficients: {coefficients}")
""",
        "Statsmodels": """
import statsmodels.api as sm

# Add constant for intercept
X_with_const = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X_with_const).fit()

# Print detailed summary
print(model.summary())

# Access specific statistics
print(f"R-squared: {model.rsquared:.4f}")
print(f"Adj. R-squared: {model.rsquared_adj:.4f}")
print(f"AIC: {model.aic:.2f}")
print(f"BIC: {model.bic:.2f}")
""",
        "Regularization (Ridge/Lasso)": """
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score

# Ridge Regression (L2 regularization)
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_score = ridge.score(X_test, y_test)

# Lasso Regression (L1 regularization)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
lasso_score = lasso.score(X_test, y_test)

print(f"Ridge R²: {ridge_score:.4f}")
print(f"Lasso R²: {lasso_score:.4f}")
print(f"Lasso coefficients (sparse): {lasso.coef_}")
"""
    })
    
    # 4. Real-World Applications (in tabs)
    st.header("💼 Real-World Applications")
    
    applications_data = {
        "House Price Prediction": {
            "description": "Predict housing prices based on features like size, location, number of bedrooms, etc.",
            "details": "Used by real estate platforms, mortgage lenders, and property investors to estimate market values and make informed decisions.",
            "examples": [
                "Zillow's Zestimate algorithm",
                "Redfin's home value estimates",
                "Bank mortgage approval systems"
            ]
        },
        "Sales Forecasting": {
            "description": "Forecast future sales based on historical data, marketing spend, and economic indicators.",
            "details": "Helps businesses optimize inventory, plan marketing campaigns, and allocate resources effectively.",
            "examples": [
                "Retail chain sales predictions",
                "E-commerce demand forecasting",
                "Subscription service revenue projections"
            ]
        },
        "Risk Assessment": {
            "description": "Assess financial or insurance risk based on customer demographics and behavior.",
            "details": "Used by banks, insurance companies, and financial institutions to quantify risk exposure.",
            "examples": [
                "Credit scoring for loan applications",
                "Insurance premium calculation",
                "Investment risk analysis"
            ]
        },
        "Medical Research": {
            "description": "Model relationships between clinical measurements and health outcomes.",
            "details": "Helps researchers identify risk factors, predict disease progression, and evaluate treatment effectiveness.",
            "examples": [
                "Predicting patient recovery time",
                "Modeling drug dosage effects",
                "Identifying disease risk factors"
            ]
        }
    }
    
    app_tabs = st.tabs(list(applications_data.keys()))
    
    for tab, (app_name, app_info) in zip(app_tabs, applications_data.items()):
        with tab:
            st.subheader(app_name)
            st.markdown(f"**Description:** {app_info['description']}")
            st.markdown(f"**Details:** {app_info['details']}")
            st.markdown("**Examples:**")
            for example in app_info['examples']:
                st.markdown(f"- {example}")
    
    # 5. Common Pitfalls & Fixes (in tabs)
    st.header("⚠️ Common Pitfalls & Fixes")
    
    pitfalls_data = {
        "Multicollinearity": {
            "problem": "High correlation between independent variables can make coefficient estimates unstable and difficult to interpret.",
            "detection": "Calculate Variance Inflation Factor (VIF) - values above 5-10 indicate multicollinearity.",
            "solution": "Remove correlated features, use regularization (Ridge/Lasso), or apply Principal Component Analysis (PCA)."
        },
        "Overfitting": {
            "problem": "Model fits training data too closely and performs poorly on new, unseen data.",
            "detection": "Large gap between training and test performance metrics.",
            "solution": "Use train-test split, cross-validation, regularization, or reduce model complexity."
        },
        "Non-linearity": {
            "problem": "Assuming linear relationship when true relationship is non-linear.",
            "detection": "Check residual plots for patterns, use polynomial feature testing.",
            "solution": "Add polynomial features, use non-linear models, or apply transformations."
        },
        "Heteroscedasticity": {
            "problem": "Non-constant variance of errors across predictions.",
            "detection": "Fan-shaped pattern in residual plots, Breusch-Pagan test.",
            "solution": "Transform variables (log, sqrt), use weighted least squares, or robust regression."
        },
        "Outliers": {
            "problem": "Extreme values can disproportionately influence the regression line.",
            "detection": "Cook's distance, leverage plots, studentized residuals.",
            "solution": "Remove influential points, use robust regression methods, or apply transformations."
        }
    }
    
    pitfall_tabs = st.tabs(list(pitfalls_data.keys()))
    
    for tab, (pitfall_name, pitfall_info) in zip(pitfall_tabs, pitfalls_data.items()):
        with tab:
            st.subheader(pitfall_name)
            st.markdown(f"**Problem:** {pitfall_info['problem']}")
            st.markdown(f"**How to Detect:** {pitfall_info['detection']}")
            st.markdown(f"**Solution:** {pitfall_info['solution']}")
    
    # 6. References & Further Reading (in expander)
    st.header("📚 References & Further Reading")
    
    with st.expander("Click to view references", expanded=False):
        display_references(get_regression_references())
    
    # Footer
    st.markdown("---")
    st.caption(
        "Linear Regression Concept • "
        "Use the demo section to experiment with different parameters • "
        "Next: Try the Clustering page for unsupervised learning"
    )


if __name__ == "__main__":
    main()