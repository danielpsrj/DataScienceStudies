"""
General data science data shared across multiple pages.
"""

from typing import List, Dict


def get_general_data_science_references() -> List[Dict[str, str]]:
    """Get general data science references."""
    return [
        {
            "type": "book",
            "authors": "Wickham, H., & Grolemund, G.",
            "year": "2016",
            "title": "R for Data Science: Import, Tidy, Transform, Visualize, and Model Data",
            "publisher": "O'Reilly Media",
            "url": "https://r4ds.had.co.nz/",
            "abstract": "This book will teach you how to do data science with R: You'll learn how to get your data into R, get it into the most useful structure, transform it, visualise it and model it.",
            "tags": ["R", "tidyverse", "data visualization"],
        },
        {
            "type": "book",
            "authors": "McKinney, W.",
            "year": "2017",
            "title": "Python for Data Analysis: Data Wrangling with Pandas, NumPy, and IPython",
            "publisher": "O'Reilly Media",
            "url": "https://wesmckinney.com/book/",
            "abstract": "This book is concerned with the nuts and bolts of manipulating, processing, cleaning, and crunching data in Python. It is also a practical, modern introduction to scientific computing in Python.",
            "tags": ["python", "pandas", "data wrangling"],
        },
        {
            "type": "online",
            "authors": "Various Contributors",
            "year": "2023",
            "title": "Towards Data Science",
            "publisher": "Medium",
            "url": "https://towardsdatascience.com/",
            "abstract": "A Medium publication sharing concepts, ideas, and codes in data science, machine learning, and AI.",
            "tags": ["blog", "tutorials", "community"],
        },
        {
            "type": "online",
            "authors": "Kaggle Team",
            "year": "2023",
            "title": "Kaggle Learn",
            "publisher": "Kaggle",
            "url": "https://www.kaggle.com/learn",
            "abstract": "Free micro-courses on data science and machine learning topics, with hands-on exercises and competitions.",
            "tags": ["courses", "hands-on", "competitions"],
        },
    ]


def get_time_series_applications() -> List[Dict[str, str]]:
    """Get real-world applications for time series analysis."""
    return [
        {
            "title": "Stock Market Prediction",
            "description": "Forecast stock prices and market trends based on historical data.",
            "details": "Used by traders, investors, and financial institutions for decision making.",
            "examples": [
                "Daily stock price forecasting",
                "Market volatility prediction",
                "Trading signal generation",
            ],
        },
        {
            "title": "Energy Demand Forecasting",
            "description": "Predict electricity consumption patterns for grid management.",
            "details": "Helps utility companies optimize generation, reduce costs, and prevent blackouts.",
            "examples": [
                "Hourly electricity demand prediction",
                "Renewable energy output forecasting",
                "Peak load management",
            ],
        },
        {
            "title": "Weather Forecasting",
            "description": "Predict meteorological conditions based on historical patterns.",
            "details": "Critical for agriculture, transportation, disaster management, and daily planning.",
            "examples": [
                "Temperature and precipitation forecasts",
                "Hurricane tracking and intensity prediction",
                "Seasonal climate patterns",
            ],
        },
        {
            "title": "Website Traffic Analysis",
            "description": "Analyze and predict web traffic patterns for capacity planning.",
            "details": "Helps websites optimize server resources, plan marketing campaigns, and improve user experience.",
            "examples": [
                "Daily visitor prediction",
                "Peak traffic hour analysis",
                "Seasonal trend identification",
            ],
        },
    ]


def get_classification_pitfalls() -> List[Dict[str, str]]:
    """Get common pitfalls for classification analysis."""
    return [
        {
            "title": "Class Imbalance",
            "description": "When one class has significantly more samples than others, leading to biased models.",
            "severity": "High",
            "solution": [
                "Use resampling techniques (oversampling minority, undersampling majority)",
                "Apply class weights in algorithms",
                "Use appropriate evaluation metrics (precision, recall, F1-score)",
                "Try anomaly detection approaches for extreme imbalance",
            ],
            "tips": "Accuracy is misleading with imbalanced data. Always use metrics like precision, recall, and F1-score.",
        },
        {
            "title": "Data Leakage",
            "description": "Allowing information from test/validation data to influence training, leading to overly optimistic results.",
            "severity": "High",
            "solution": [
                "Always split data before any preprocessing",
                "Use pipeline objects to prevent leakage",
                "Be careful with time-series data (use time-based splits)",
                "Validate preprocessing steps are fit only on training data",
            ],
            "tips": "If your model performs suspiciously well, check for data leakage. This is a common cause of unrealistic performance.",
        },
        {
            "title": "Ignoring Probability Calibration",
            "description": "Treating model probabilities as true probabilities without calibration.",
            "severity": "Medium",
            "solution": [
                "Use calibration curves to assess probability quality",
                "Apply Platt scaling or isotonic regression",
                "Choose algorithms with naturally calibrated probabilities (like logistic regression)",
                "Use ensemble methods with probability calibration",
            ],
            "tips": "Some algorithms (like SVM, decision trees) produce poorly calibrated probabilities. Always check calibration if using probabilities for decision making.",
        },
    ]