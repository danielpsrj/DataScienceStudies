"""
Interactive demo component for concept pages.
"""
from typing import Any, Callable, Dict, List, Optional, Union
import streamlit as st

from app.state import get_state


def interactive_demo(
    title: str,
    param_definitions: List[Dict[str, Any]],
    run_callback: Callable,
    results_callback: Optional[Callable] = None,
    icon: str = "🎮",
    show_code: bool = True,
    code_snippet: Optional[str] = None,
) -> None:
    """
    Standard interactive demo with parameter controls.
    
    Args:
        title: Demo title
        param_definitions: List of dicts with parameter definitions
        run_callback: Function that receives parameter values and returns results
        results_callback: Optional function to display results
        icon: Icon to display before title
        show_code: Whether to show code snippet
        code_snippet: Optional code snippet to display
    """
    st.header(f"{icon} {title}")
    
    # Get user preferences
    state = get_state()
    auto_run = state.get_preference("auto_run", False)
    
    # Parameter controls in sidebar
    with st.sidebar:
        st.subheader("⚙️ Parameters")
        params = {}
        
        for param in param_definitions:
            param_type = param.get("type", "slider")
            param_name = param.get("name")
            param_args = param.get("args", {})
            
            if param_type == "slider":
                params[param_name] = st.slider(**param_args)
            elif param_type == "selectbox":
                params[param_name] = st.selectbox(**param_args)
            elif param_type == "checkbox":
                params[param_name] = st.checkbox(**param_args)
            elif param_type == "number_input":
                params[param_name] = st.number_input(**param_args)
            elif param_type == "text_input":
                params[param_name] = st.text_input(**param_args)
            elif param_type == "radio":
                params[param_name] = st.radio(**param_args)
            else:
                st.warning(f"Unknown parameter type: {param_type}")
    
    # Run button and results
    run_button = st.button("🚀 Run Demo", type="primary", disabled=auto_run)
    
    if auto_run or run_button:
        with st.spinner("Running demo..."):
            try:
                results = run_callback(**params)
                
                # Display results
                if results_callback:
                    results_callback(results, params)
                else:
                    _default_results_display(results, params)
                
                st.success("Demo completed successfully!")
                
            except Exception as e:
                st.error(f"Error running demo: {str(e)}")
                if st.button("Show Traceback"):
                    st.exception(e)
    
    # Code snippet
    if show_code and code_snippet:
        with st.expander("💻 Demo Code", expanded=False):
            st.code(code_snippet, language="python")
    
    # Parameter summary
    with st.expander("📋 Parameter Summary", expanded=False):
        st.markdown("**Current Parameters:**")
        for param_name, param_value in params.items():
            st.markdown(f"- `{param_name}`: `{param_value}`")


def _default_results_display(results: Any, params: Dict[str, Any]) -> None:
    """Default results display if no custom callback provided."""
    if isinstance(results, dict):
        # Display dictionary as metrics or JSON
        cols = st.columns(min(4, len(results)))
        
        for idx, (key, value) in enumerate(results.items()):
            col = cols[idx % len(cols)]
            with col:
                st.metric(label=key, value=f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
        
        # Show raw results in expander
        with st.expander("📊 Raw Results"):
            st.json(results)
    
    elif hasattr(results, "shape"):  # NumPy array or pandas DataFrame
        try:
            import pandas as pd
            if isinstance(results, pd.DataFrame):
                st.dataframe(results)
            else:
                # Convert array to DataFrame for display
                df = pd.DataFrame(results)
                st.dataframe(df)
        except ImportError:
            st.write(results)
    
    else:
        # Try to display as string or object
        st.write(results)


def comparison_demo(
    title: str,
    algorithms: List[Dict[str, Any]],
    data_generator: Callable,
    evaluator: Callable,
    param_definitions: Optional[List[Dict[str, Any]]] = None,
    icon: str = "📊",
) -> None:
    """
    Demo that compares multiple algorithms.
    
    Args:
        title: Demo title
        algorithms: List of algorithm definitions
        data_generator: Function that generates data
        evaluator: Function that evaluates algorithm results
        param_definitions: Optional shared parameter definitions
        icon: Icon to display before title
    """
    st.header(f"{icon} {title}")
    
    # Algorithm selection
    algorithm_names = [alg["name"] for alg in algorithms]
    selected_alg = st.selectbox("Select Algorithm", algorithm_names)
    
    # Get selected algorithm
    selected_algorithm = next(alg for alg in algorithms if alg["name"] == selected_alg)
    
    # Algorithm-specific parameters
    with st.sidebar:
        st.subheader(f"⚙️ {selected_alg} Parameters")
        
        alg_params = {}
        for param in selected_algorithm.get("parameters", []):
            param_type = param.get("type", "slider")
            param_name = param.get("name")
            param_args = param.get("args", {})
            
            if param_type == "slider":
                alg_params[param_name] = st.slider(**param_args)
            elif param_type == "selectbox":
                alg_params[param_name] = st.selectbox(**param_args)
            elif param_type == "checkbox":
                alg_params[param_name] = st.checkbox(**param_args)
    
    # Shared parameters
    shared_params = {}
    if param_definitions:
        with st.sidebar:
            st.subheader("⚙️ Shared Parameters")
            
            for param in param_definitions:
                param_type = param.get("type", "slider")
                param_name = param.get("name")
                param_args = param.get("args", {})
                
                if param_type == "slider":
                    shared_params[param_name] = st.slider(**param_args)
                elif param_type == "selectbox":
                    shared_params[param_name] = st.selectbox(**param_args)
                elif param_type == "checkbox":
                    shared_params[param_name] = st.checkbox(**param_args)
    
    # Run comparison
    if st.button("🚀 Run Comparison", type="primary"):
        with st.spinner("Running comparison..."):
            # Generate data
            data = data_generator(**shared_params)
            
            # Run all algorithms
            results = {}
            for algorithm in algorithms:
                alg_name = algorithm["name"]
                alg_function = algorithm["function"]
                alg_params = algorithm.get("default_params", {})
                
                try:
                    result = alg_function(data, **alg_params)
                    results[alg_name] = result
                except Exception as e:
                    st.warning(f"Error running {alg_name}: {str(e)}")
                    results[alg_name] = None
            
            # Evaluate and display results
            evaluation = evaluator(results, data)
            
            # Display comparison
            st.subheader("📈 Comparison Results")
            
            if isinstance(evaluation, dict):
                # Convert to DataFrame for better display
                try:
                    import pandas as pd
                    df = pd.DataFrame.from_dict(evaluation, orient="index")
                    st.dataframe(df.style.highlight_max(axis=0))
                except ImportError:
                    st.json(evaluation)
            
            # Show individual results
            with st.expander("🔍 Individual Algorithm Results"):
                for alg_name, result in results.items():
                    if result is not None:
                        st.subheader(f"{alg_name}")
                        st.write(result)


# Example usage:
"""
# In a concept page:

from app.components import interactive_demo, comparison_demo
import numpy as np
from sklearn.linear_model import LinearRegression

# Simple interactive demo
def run_regression_demo(sample_size, noise_level):
    # Generate data
    X = np.random.randn(sample_size, 1)
    y = 2 * X.squeeze() + np.random.normal(0, noise_level, sample_size)
    
    # Train model
    model = LinearRegression()
    model.fit(X, y)
    
    return {
        "coefficient": model.coef_[0],
        "intercept": model.intercept_,
        "r2_score": model.score(X, y)
    }

interactive_demo(
    title="Linear Regression Demo",
    param_definitions=[
        {
            "name": "sample_size",
            "type": "slider",
            "args": {"label": "Sample Size", "min": 10, "max": 1000, "value": 100}
        },
        {
            "name": "noise_level",
            "type": "slider", 
            "args": {"label": "Noise Level", "min": 0.0, "max": 1.0, "value": 0.2}
        }
    ],
    run_callback=run_regression_demo,
    code_snippet=\"\"\"
def run_regression_demo(sample_size, noise_level):
    X = np.random.randn(sample_size, 1)
    y = 2 * X.squeeze() + np.random.normal(0, noise_level, sample_size)
    
    model = LinearRegression()
    model.fit(X, y)
    
    return {
        "coefficient": model.coef_[0],
        "intercept": model.intercept_,
        "r2_score": model.score(X, y)
    }
\"\"\"
)
"""