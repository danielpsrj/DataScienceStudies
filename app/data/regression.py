"""
Regression-specific data for the Linear Regression concept page.
"""

from typing import List, Dict


def get_regression_applications() -> List[Dict[str, str]]:
    """Get real-world applications for regression analysis."""
    return [
        {
            "title": "Sales Forecasting",
            "description": "Predict future sales based on historical data, marketing spend, and economic indicators.",
            "details": "Used by retail companies to optimize inventory, plan promotions, and allocate resources.",
            "examples": [
                "Predicting monthly sales for a retail chain",
                "Forecasting demand for seasonal products",
                "Estimating revenue growth for startups",
            ],
        },
        {
            "title": "Risk Assessment",
            "description": "Quantify risk factors in finance, insurance, and healthcare.",
            "details": "Helps organizations make data-driven decisions about risk exposure and mitigation strategies.",
            "examples": [
                "Credit scoring for loan applications",
                "Insurance premium calculation",
                "Medical risk prediction",
            ],
        },
        {
            "title": "Price Optimization",
            "description": "Determine optimal pricing based on market conditions, competition, and customer behavior.",
            "details": "Used by e-commerce platforms, airlines, and hospitality businesses to maximize revenue.",
            "examples": [
                "Dynamic pricing for ride-sharing services",
                "Hotel room pricing based on demand",
                "E-commerce product pricing algorithms",
            ],
        },
        {
            "title": "Quality Control",
            "description": "Monitor and predict product quality in manufacturing processes.",
            "details": "Helps identify factors affecting quality and optimize production parameters.",
            "examples": [
                "Predicting defect rates in manufacturing",
                "Optimizing chemical process parameters",
                "Monitoring equipment performance",
            ],
        },
    ]


def get_regression_pitfalls() -> List[Dict[str, str]]:
    """Get common pitfalls for regression analysis."""
    return [
        {
            "title": "Overfitting",
            "description": "Creating a model that fits the training data too closely, capturing noise rather than underlying patterns.",
            "severity": "High",
            "solution": [
                "Use cross-validation to evaluate model performance",
                "Apply regularization techniques (L1/L2 regularization)",
                "Simplify the model by reducing features",
                "Use more training data when possible",
            ],
            "tips": "Always validate your model on unseen data. A model that performs perfectly on training data but poorly on test data is likely overfit.",
        },
        {
            "title": "Multicollinearity",
            "description": "When predictor variables are highly correlated with each other, making it difficult to determine individual effects.",
            "severity": "Medium",
            "solution": [
                "Calculate Variance Inflation Factor (VIF) to detect multicollinearity",
                "Remove highly correlated features",
                "Use dimensionality reduction techniques (PCA)",
                "Apply regularization methods",
            ],
            "tips": "Check correlation matrices before building models. Features with correlation > 0.8 or < -0.8 should be investigated.",
        },
        {
            "title": "Ignoring Non-Linear Relationships",
            "description": "Assuming linear relationships when the true relationship is non-linear.",
            "severity": "Medium",
            "solution": [
                "Plot residuals to check for patterns",
                "Use polynomial features or interaction terms",
                "Consider non-linear models (decision trees, neural networks)",
                "Apply transformations (log, square root) to variables",
            ],
            "tips": "Always visualize the relationship between variables before assuming linearity. Scatter plots with trend lines can reveal non-linear patterns.",
        },
        {
            "title": "Outliers Unduly Influencing Results",
            "description": "Extreme values disproportionately affecting regression coefficients and predictions.",
            "severity": "High",
            "solution": [
                "Identify outliers using box plots or statistical methods (IQR)",
                "Consider robust regression methods",
                "Transform variables to reduce outlier impact",
                "Remove or winsorize outliers when justified",
            ],
            "tips": "Be cautious when removing outliers - understand why they exist before deciding to exclude them.",
        },
        {
            "title": "Ignoring Assumptions",
            "description": "Failing to check regression assumptions (linearity, independence, homoscedasticity, normality).",
            "severity": "Medium",
            "solution": [
                "Check residual plots for patterns",
                "Test for heteroscedasticity (Breusch-Pagan test)",
                "Check for autocorrelation (Durbin-Watson test)",
                "Test normality of residuals (Q-Q plots, Shapiro-Wilk test)",
            ],
            "tips": "Always validate assumptions before interpreting results. Violated assumptions can lead to incorrect conclusions.",
        },
    ]


def get_regression_references() -> List[Dict[str, str]]:
    """Get references for regression analysis."""
    return [
        {
            "type": "book",
            "authors": "James, G., Witten, D., Hastie, T., & Tibshirani, R.",
            "year": "2013",
            "title": "An Introduction to Statistical Learning: with Applications in R",
            "publisher": "Springer",
            "url": "https://www.statlearning.com/",
            "doi": "10.1007/978-1-4614-7138-7",
            "abstract": "This book provides an introduction to statistical learning methods. It is aimed for upper level undergraduate students, masters students and PhD students in the non-mathematical sciences. The book also contains a number of R labs with detailed explanations on how to implement the various methods in real life settings.",
            "tags": ["introductory", "R", "statistical learning"],
        },
        {
            "type": "book",
            "authors": "Hastie, T., Tibshirani, R., & Friedman, J.",
            "year": "2009",
            "title": "The Elements of Statistical Learning: Data Mining, Inference, and Prediction",
            "publisher": "Springer",
            "url": "https://hastie.su.domains/ElemStatLearn/",
            "doi": "10.1007/978-0-387-84858-7",
            "abstract": "This book describes the important ideas in these areas in a common conceptual framework. While the approach is statistical, the emphasis is on concepts rather than mathematics. Many examples are given, with a liberal use of color graphics.",
            "tags": ["advanced", "machine learning", "data mining"],
        },
        {
            "type": "paper",
            "authors": "Tibshirani, R.",
            "year": "1996",
            "title": "Regression Shrinkage and Selection via the Lasso",
            "journal": "Journal of the Royal Statistical Society: Series B (Methodological)",
            "url": "https://www.jstor.org/stable/2346178",
            "doi": "10.1111/j.2517-6161.1996.tb02080.x",
            "abstract": "We propose a new method for estimation in linear models. The 'lasso' minimizes the residual sum of squares subject to the sum of the absolute value of the coefficients being less than a constant.",
            "tags": ["lasso", "regularization", "feature selection"],
        },
        {
            "type": "article",
            "authors": "Hoerl, A. E., & Kennard, R. W.",
            "year": "1970",
            "title": "Ridge Regression: Biased Estimation for Nonorthogonal Problems",
            "journal": "Technometrics",
            "url": "https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634",
            "doi": "10.1080/00401706.1970.10488634",
            "abstract": "Ridge regression is a way to create a parsimonious model when the number of predictor variables in a set exceeds the number of observations, or when a data set exhibits multicollinearity.",
            "tags": ["ridge regression", "multicollinearity", "regularization"],
        },
        {
            "type": "online",
            "authors": "Scikit-learn Developers",
            "year": "2023",
            "title": "Linear Models - scikit-learn documentation",
            "publisher": "scikit-learn",
            "url": "https://scikit-learn.org/stable/modules/linear_model.html",
            "abstract": "Documentation for linear models in scikit-learn, including ordinary least squares, ridge regression, lasso, and elastic net.",
            "tags": ["python", "scikit-learn", "documentation"],
        },
    ]