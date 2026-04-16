"""
Linear Regression concept page.
"""
import streamlit as st
import numpy as np
import plotly.graph_objects as go

from app.components import (
    theory_section,
    math_equation,
    interactive_demo,
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
    
    # 1. Theory section
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
    
    # 2. Mathematical formulation
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
        expandable=True,
    )
    
    # Matrix formulation
    with st.expander("📐 Matrix Formulation", expanded=False):
        st.latex(r"\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}")
        st.markdown("**Where:**")
        st.markdown(r"- $\mathbf{y}$: Vector of target values $(n \times 1)$")
        st.markdown(r"- $\mathbf{X}$: Design matrix $(n \times p)$")
        st.markdown(r"- $\boldsymbol{\beta}$: Coefficient vector $(p \times 1)$")
        st.markdown(r"- $\boldsymbol{\epsilon}$: Error vector $(n \times 1)$")
        
        st.markdown("**Ordinary Least Squares (OLS) Solution:**")
        st.latex(r"\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}")
    
    # 3. Interactive demo
    st.header("🎮 Interactive Demo")
    
    # Parameter controls in sidebar
    with st.sidebar:
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
    
    # Run demo button
    if st.button("🚀 Run Linear Regression", type="primary"):
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
    
    # 4. Implementation examples
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
    
    # 5. Applications
    display_applications([
        {
            "title": "House Price Prediction",
            "description": "Predict housing prices based on features like size, location, number of bedrooms, etc.",
        },
        {
            "title": "Sales Forecasting",
            "description": "Forecast future sales based on historical data, marketing spend, and economic indicators.",
        },
        {
            "title": "Risk Assessment",
            "description": "Assess financial or insurance risk based on customer demographics and behavior.",
        },
        {
            "title": "Medical Research",
            "description": "Model relationships between clinical measurements and health outcomes.",
        },
        {
            "title": "Energy Consumption",
            "description": "Predict energy usage based on weather, time of day, and building characteristics.",
        },
        {
            "title": "Academic Performance",
            "description": "Predict student grades based on study hours, attendance, and previous performance.",
        },
    ])
    
    # 6. Common pitfalls
    display_pitfalls([
        {
            "title": "Multicollinearity",
            "description": "High correlation between independent variables can make coefficient estimates unstable.",
            "solution": "Use variance inflation factor (VIF) to detect, consider regularization or feature selection."
        },
        {
            "title": "Overfitting",
            "description": "Model fits training data too closely and performs poorly on new data.",
            "solution": "Use train-test split, cross-validation, regularization, or reduce model complexity."
        },
        {
            "title": "Non-linearity",
            "description": "Assuming linear relationship when true relationship is non-linear.",
            "solution": "Check residual plots, consider polynomial features, or use non-linear models."
        },
        {
            "title": "Heteroscedasticity",
            "description": "Non-constant variance of errors across predictions.",
            "solution": "Transform variables, use weighted least squares, or robust regression methods."
        },
        {
            "title": "Outliers",
            "description": "Extreme values can disproportionately influence the regression line.",
            "solution": "Detect with Cook's distance, use robust regression, or remove influential points."
        },
        {
            "title": "Autocorrelation",
            "description": "Correlation between consecutive errors in time series data.",
            "solution": "Use Durbin-Watson test, consider time series models or include lagged variables."
        },
    ])
    
    # 7. Advanced topics
    with st.expander("🚀 Advanced Topics", expanded=False):
        st.markdown("""
        ### Regularization Techniques
        
        **Ridge Regression (L2)**
        - Adds penalty proportional to squared magnitude of coefficients
        - Helps with multicollinearity
        - All coefficients shrink but none become exactly zero
        
        **Lasso Regression (L1)**
        - Adds penalty proportional to absolute magnitude of coefficients
        - Performs feature selection (some coefficients become zero)
        - Useful for high-dimensional data
        
        **Elastic Net**
        - Combines L1 and L2 penalties
        - Balances feature selection and coefficient shrinkage
        
        ### Assumption Checking
        
        Always validate these assumptions:
        1. **Linearity**: Relationship between X and y is linear
        2. **Independence**: Observations are independent
        3. **Homoscedasticity**: Constant variance of errors
        4. **Normality**: Errors are normally distributed
        5. **No multicollinearity**: Features are not highly correlated
        
        ### Model Evaluation Metrics
        
        - **R²**: Proportion of variance explained (0 to 1, higher is better)
        - **Adjusted R²**: R² adjusted for number of predictors
        - **MSE**: Mean squared error (lower is better)
        - **RMSE**: Root mean squared error (in original units)
        - **MAE**: Mean absolute error (robust to outliers)
        """)
    
    # 8. References
    display_references(get_regression_references())
    
    # Footer
    st.markdown("---")
    st.caption(
        "Linear Regression Concept • "
        "Use the sidebar to adjust parameters and run the demo • "
        "Next: Try the Clustering page for unsupervised learning"
    )


if __name__ == "__main__":
    main()