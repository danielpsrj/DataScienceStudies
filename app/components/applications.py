"""
Applications component for data science concept pages.
Displays real-world applications of the concept being taught.
"""

import streamlit as st
from typing import List, Dict, Optional


def display_applications(
    applications: List[Dict[str, str]],
    title: str = "Real-World Applications",
    columns: int = 2,
    expand_all: bool = False,
) -> None:
    """
    Display a grid of real-world applications with descriptions.
    
    Args:
        applications: List of dictionaries with 'title' and 'description' keys
        title: Section title
        columns: Number of columns in the grid
        expand_all: Whether to expand all application cards by default
    """
    if not applications:
        st.info("No applications available for this concept.")
        return
    
    st.header(title)
    st.markdown("---")
    
    # Create columns for the grid
    cols = st.columns(columns)
    
    for i, app in enumerate(applications):
        col_idx = i % columns
        with cols[col_idx]:
            with st.expander(f"**{app['title']}**", expanded=expand_all):
                st.markdown(app['description'])
                
                # Add optional details if present
                if 'details' in app:
                    st.markdown("**Details:**")
                    st.markdown(app['details'])
                
                # Add optional examples if present
                if 'examples' in app:
                    st.markdown("**Examples:**")
                    if isinstance(app['examples'], list):
                        for example in app['examples']:
                            st.markdown(f"- {example}")
                    else:
                        st.markdown(app['examples'])


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
                "Estimating revenue growth for startups"
            ]
        },
        {
            "title": "Risk Assessment",
            "description": "Quantify risk factors in finance, insurance, and healthcare.",
            "details": "Helps organizations make data-driven decisions about risk exposure and mitigation strategies.",
            "examples": [
                "Credit scoring for loan applications",
                "Insurance premium calculation",
                "Medical risk prediction"
            ]
        },
        {
            "title": "Price Optimization",
            "description": "Determine optimal pricing based on market conditions, competition, and customer behavior.",
            "details": "Used by e-commerce platforms, airlines, and hospitality businesses to maximize revenue.",
            "examples": [
                "Dynamic pricing for ride-sharing services",
                "Hotel room pricing based on demand",
                "E-commerce product pricing algorithms"
            ]
        },
        {
            "title": "Quality Control",
            "description": "Monitor and predict product quality in manufacturing processes.",
            "details": "Helps identify factors affecting quality and optimize production parameters.",
            "examples": [
                "Predicting defect rates in manufacturing",
                "Optimizing chemical process parameters",
                "Monitoring equipment performance"
            ]
        }
    ]


def get_clustering_applications() -> List[Dict[str, str]]:
    """Get real-world applications for clustering analysis."""
    return [
        {
            "title": "Customer Segmentation",
            "description": "Group customers based on purchasing behavior, demographics, and preferences.",
            "details": "Enables targeted marketing, personalized recommendations, and improved customer retention.",
            "examples": [
                "E-commerce customer segmentation",
                "Banking customer profiling",
                "Subscription service user groups"
            ]
        },
        {
            "title": "Image Segmentation",
            "description": "Group pixels in images based on color, texture, or other features.",
            "details": "Used in computer vision for object detection, medical imaging, and autonomous vehicles.",
            "examples": [
                "Medical image analysis (tumor detection)",
                "Satellite image classification",
                "Facial recognition systems"
            ]
        },
        {
            "title": "Anomaly Detection",
            "description": "Identify unusual patterns or outliers in data.",
            "details": "Critical for fraud detection, network security, and system monitoring.",
            "examples": [
                "Credit card fraud detection",
                "Network intrusion detection",
                "Manufacturing defect detection"
            ]
        },
        {
            "title": "Document Clustering",
            "description": "Group similar documents for organization and retrieval.",
            "details": "Used in information retrieval, recommendation systems, and content management.",
            "examples": [
                "News article categorization",
                "Research paper organization",
                "Customer support ticket grouping"
            ]
        }
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
                "Trading signal generation"
            ]
        },
        {
            "title": "Energy Demand Forecasting",
            "description": "Predict electricity consumption patterns for grid management.",
            "details": "Helps utility companies optimize generation, reduce costs, and prevent blackouts.",
            "examples": [
                "Hourly electricity demand prediction",
                "Renewable energy output forecasting",
                "Peak load management"
            ]
        },
        {
            "title": "Weather Forecasting",
            "description": "Predict meteorological conditions based on historical patterns.",
            "details": "Critical for agriculture, transportation, disaster management, and daily planning.",
            "examples": [
                "Temperature and precipitation forecasts",
                "Hurricane tracking and intensity prediction",
                "Seasonal climate patterns"
            ]
        },
        {
            "title": "Website Traffic Analysis",
            "description": "Analyze and predict web traffic patterns for capacity planning.",
            "details": "Helps websites optimize server resources, plan marketing campaigns, and improve user experience.",
            "examples": [
                "Daily visitor prediction",
                "Peak traffic hour analysis",
                "Seasonal trend identification"
            ]
        }
    ]


# Example usage in a Streamlit app:
if __name__ == "__main__":
    st.title("Applications Component Demo")
    
    st.subheader("Regression Applications")
    display_applications(get_regression_applications())
    
    st.subheader("Clustering Applications")
    display_applications(get_clustering_applications(), columns=3)
    
    st.subheader("Time Series Applications (Expanded)")
    display_applications(get_time_series_applications(), expand_all=True)