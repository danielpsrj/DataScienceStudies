"""
Template for new concept pages in the Data Science Platform.

This template enforces the standardized 6-section structure:
1. Concept Overview
2. Interactive Demo
3. Implementation Examples
4. Real-World Applications
5. Common Pitfalls & Fixes
6. References & Further Reading

IMPORTANT FOR AI AGENTS:
- Follow this exact structure for all new concept pages
- Use the existing components from app.components
- Keep demo controls in columns within the demo section (not sidebar)
- Present applications and pitfalls in interactive tabs
- Put references section inside expander
- Always add comprehensive tests for new functionality
"""

import streamlit as st

from app.components import (
    theory_section,
    math_equation,
    code_tabs,
    # Import specific references for your concept
    # from app.components.references import get_[concept]_references
)

# from app.logic.[concept_module] import (
#     # Import your concept-specific functions here
#     # Example: generate_data, train_model, visualize_results, etc.
# )
from app.state import get_state


def main() -> None:
    """Main page function - follow this structure exactly."""
    # ==================== PAGE HEADER ====================
    st.title("[ICON] [Concept Name]")
    st.caption("[Brief description of the concept]")

    # Track page visit
    state = get_state()
    state.add_to_history("[concept_name]")
    state.current_model = "[concept_name]"

    # ==================== 1. CONCEPT OVERVIEW ====================
    theory_section(
        title="Concept Overview",
        content="""
        **[Concept Name]** is a [type of algorithm/technique] that [brief description].
        
        ### Key Characteristics:
        - **Characteristic 1**: Description
        - **Characteristic 2**: Description
        - **Characteristic 3**: Description
        - **Characteristic 4**: Description
        
        [Additional detailed explanation of the concept, its importance, and use cases.]
        """,
        image_url="[URL to relevant image or diagram]",
        columns=(2, 1),  # Adjust columns as needed
    )

    # Mathematical formulation in expander (for longer content)
    with st.expander("🧮 Mathematical Formulation", expanded=False):
        math_equation(
            equation=r"[LaTeX equation]",
            variables={
                "[variable1]": "Description",
                "[variable2]": "Description",
                "[variable3]": "Description",
            },
            title="Mathematical Formulation",
            icon="📐",
            expandable=False,
        )

        # Additional mathematical details if needed
        # st.markdown("**Additional Formulation:**")
        # st.latex(r"[additional LaTeX]")
        # st.markdown("**Where:**")
        # st.markdown(r"- $\mathbf{[symbol]}$: Description")

    # ==================== 2. INTERACTIVE DEMO ====================
   
    with st.expander("🎮 Interactive Demo", expanded=True):
    # Demo controls at the top (horizontal layout)
    
    # Create horizontal columns for controls
    # Adjust number of columns based on your parameters
        control_col1, control_col2, control_col3, control_col4 = st.columns(4)

        with control_col1:
            # Add parameter controls here
            # Example:
            # param1 = st.slider(
            #     "Parameter 1",
            #     min_value=0,
            #     max_value=100,
            #     value=50,
            #     step=1,
            #     help="Description of parameter 1"
            # )
            pass

        with control_col2:
            # Example:
            # param2 = st.selectbox(
            #     "Algorithm Type",
            #     ["Option 1", "Option 2", "Option 3"],
            #     index=0,
            #     help="Description of algorithm selection"
            # )
            pass

        with control_col3:
            # Example:
            # param3 = st.slider(
            #     "Parameter 3",
            #     min_value=0.0,
            #     max_value=1.0,
            #     value=0.5,
            #     step=0.1,
            #     help="Description of parameter 3"
            # )
            pass

        with control_col4:
            # Example:
            # show_details = st.checkbox(
            #     "Show Detailed Results",
            #     value=False,
            #     help="Display additional analysis"
            # )
            pass

        # Run button centered below controls
        run_col1, run_col2, run_col3 = st.columns([1, 2, 1])
        with run_col2:
            if st.button(
                "🚀 Run [Concept] Analysis", type="primary", use_container_width=True
            ):
                with st.spinner("Running analysis..."):
                    # Generate data and run analysis
                    # Example:
                    # data = generate_data(param1, param2, param3)
                    # results = analyze_data(data)

                    # Display results
                    # col1, col2 = st.columns(2)
                    # with col1:
                    #     st.metric("Metric 1", f"{results['metric1']:.4f}")
                    #     st.metric("Metric 2", f"{results['metric2']:.4f}")
                    # with col2:
                    #     st.metric("Metric 3", f"{results['metric3']:.4f}")
                    #     st.metric("Metric 4", f"{results['metric4']:.4f}")

                    # Visualize results
                    # fig = visualize_results(results, show_details=show_details)
                    # st.plotly_chart(fig, use_container_width=True)

                    # Show raw data if applicable
                    # with st.expander("📊 Generated Data", expanded=False):
                    #     import pandas as pd
                    #     df = pd.DataFrame(data)
                    #     st.dataframe(df.head(10))
                    #     st.caption(f"Showing 10 of {len(df)} rows")

                    st.success("Analysis complete!")

    # ==================== 3. IMPLEMENTATION EXAMPLES ====================
    with st.expander("💻 Implementation Examples", expanded=True):
        code_tabs(
        {
            "Library 1 Implementation": """
# Example implementation using Library 1
import library1

# Code example here
def example_function(param1, param2):
    \"\"\"Example function documentation.\"\"\"
    result = library1.process(param1, param2)
    return result

# Usage example
result = example_function(value1, value2)
print(f"Result: {result}")
""",
            "Library 2 Implementation": """
# Alternative implementation using Library 2
import library2
import numpy as np

# Different approach example
def alternative_implementation(data):
    \"\"\"Alternative implementation documentation.\"\"\"
    model = library2.Model()
    model.fit(data)
    predictions = model.predict(data)
    return predictions

# Usage with sample data
sample_data = np.random.randn(100, 5)
predictions = alternative_implementation(sample_data)
""",
            "Custom Implementation": """
# Custom implementation from scratch
import numpy as np
from typing import List, Tuple

def custom_algorithm(X: np.ndarray, **kwargs) -> Tuple[np.ndarray, dict]:
    \"\"\"Custom implementation with type hints and documentation.\"\"\"
    # Implementation logic here
    results = np.mean(X, axis=0)
    metrics = {"mean": np.mean(results), "std": np.std(results)}
    return results, metrics

# Example usage
data = np.random.randn(100, 3)
results, metrics = custom_algorithm(data)
print(f"Results shape: {results.shape}")
print(f"Metrics: {metrics}")
""",
            "Advanced Usage": """
# Advanced usage with optimization
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

# Example of hyperparameter tuning
param_grid = {
    'param1': [0.1, 1.0, 10.0],
    'param2': [1, 2, 3],
    'param3': ['option1', 'option2']
}

# Grid search example
grid_search = GridSearchCV(
    estimator=YourModel(),
    param_grid=param_grid,
    scoring=make_scorer(your_scoring_function),
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")
""",
        }
    )

    # ==================== 4. REAL-WORLD APPLICATIONS (in expander with tabs inside) ====================
    with st.expander("💼 Real-World Applications", expanded=False):
        applications_data = {
            "Application 1": {
                "description": "Brief description of application 1.",
                "details": "Detailed explanation of how this concept is applied in real-world scenarios.",
                "examples": [
                    "Example 1: Specific use case",
                    "Example 2: Industry application",
                    "Example 3: Research application",
                ],
            },
            "Application 2": {
                "description": "Brief description of application 2.",
                "details": "Detailed explanation including challenges and solutions.",
                "examples": [
                    "Example 1: Business use case",
                    "Example 2: Scientific application",
                    "Example 3: Engineering problem",
                ],
            },
            "Application 3": {
                "description": "Brief description of application 3.",
                "details": "Explanation of unique aspects of this application.",
                "examples": [
                    "Example 1: Commercial product",
                    "Example 2: Academic research",
                    "Example 3: Government project",
                ],
            },
            "Application 4": {
                "description": "Brief description of application 4.",
                "details": "Future trends and emerging applications.",
                "examples": [
                    "Example 1: Cutting-edge research",
                    "Example 2: Innovative startup",
                    "Example 3: Industry 4.0 application",
                ],
            },
        }

        # Applications in tabs - FOLLOW THIS PATTERN
        app_tabs = st.tabs(list(applications_data.keys()))

        for tab, (app_name, app_info) in zip(app_tabs, applications_data.items()):
            with tab:
                st.subheader(app_name)
                st.markdown(f"**Description:** {app_info['description']}")
                st.markdown(f"**Details:** {app_info['details']}")
                st.markdown("**Examples:**")
                for example in app_info["examples"]:
                    st.markdown(f"- {example}")

    # ==================== 5. COMMON PITFALLS & FIXES (in expander with tabs inside) ====================
    with st.expander("⚠️ Common Pitfalls & Fixes", expanded=False):
        pitfalls_data = {
            "Pitfall 1": {
                "problem": "Description of the common problem or mistake.",
                "detection": "How to identify if you're experiencing this pitfall.",
                "solution": "Step-by-step solution to fix or avoid this pitfall.",
            },
            "Pitfall 2": {
                "problem": "Description of another common issue.",
                "detection": "Diagnostic methods or indicators.",
                "solution": "Best practices and solutions.",
            },
            "Pitfall 3": {
                "problem": "Description of technical challenge.",
                "detection": "Performance metrics or error patterns.",
                "solution": "Optimization techniques or alternative approaches.",
            },
            "Pitfall 4": {
                "problem": "Description of implementation error.",
                "detection": "Debugging steps or validation methods.",
                "solution": "Correct implementation with examples.",
            },
            "Pitfall 5": {
                "problem": "Description of conceptual misunderstanding.",
                "detection": "Educational resources or diagnostic tests.",
                "solution": "Clarification with analogies or examples.",
            },
        }

        # Pitfalls in tabs - FOLLOW THIS PATTERN
        pitfall_tabs = st.tabs(list(pitfalls_data.keys()))

        for tab, (pitfall_name, pitfall_info) in zip(
            pitfall_tabs, pitfalls_data.items()
        ):
            with tab:
                st.subheader(pitfall_name)
                st.markdown(f"**Problem:** {pitfall_info['problem']}")
                st.markdown(f"**How to Detect:** {pitfall_info['detection']}")
                st.markdown(f"**Solution:** {pitfall_info['solution']}")

    # ==================== 6. REFERENCES & FURTHER READING ====================
    with st.expander("📚 References & Further Reading", expanded=False):
        # Use appropriate references function for your concept
        # Example: display_references(get_[concept]_references())
        st.markdown("""
        ### Books
        - **Book Title 1** by Author Name - Comprehensive coverage of [concept]
        - **Book Title 2** by Author Name - Practical applications and examples
        
        ### Research Papers
        - **Paper Title 1** (Year) - Foundational paper on [concept]
        - **Paper Title 2** (Year) - Recent advances in [concept]
        
        ### Online Resources
        - [Official Documentation](https://example.com) - Library documentation
        - [Tutorial Series](https://example.com/tutorials) - Step-by-step guides
        - [Interactive Course](https://example.com/course) - Hands-on learning
        
        ### Community Resources
        - [GitHub Repository](https://github.com/example) - Reference implementations
        - [Stack Overflow Tag](https://stackoverflow.com/questions/tagged/[concept]) - Q&A
        - [Discussion Forum](https://forum.example.com) - Community discussions
        """)

    # ==================== FOOTER ====================
    st.markdown("---")
    st.caption(
        "[Concept Name] Concept • "
        "Use the demo section to experiment with different parameters • "
        "Next: Explore other data science concepts in the platform"
    )


if __name__ == "__main__":
    main()
