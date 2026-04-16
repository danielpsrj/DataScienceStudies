"""
Main entry point for the Streamlit Data Science Platform.
This file sets up the application configuration and serves as the router.
"""
import streamlit as st

from app.config import settings
from app.state import get_state


def setup_page_config() -> None:
    """Set up Streamlit page configuration."""
    st.set_page_config(
        page_title=settings.app_name,
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/yourusername/data-science-platform/issues",
            "Report a bug": "https://github.com/yourusername/data-science-platform/issues",
            "About": f"""
            # {settings.app_name}
            
            A platform for exploring data science concepts with interactive examples.
            
            **Version:** 0.1.0  
            **Environment:** {settings.app_env}  
            **Debug:** {settings.debug}
            """,
        },
    )


def setup_sidebar() -> None:
    """Set up the application sidebar."""
    with st.sidebar:
        st.title(f"🧪 {settings.app_name}")
        st.markdown("---")
        
        # Environment indicator
        env_color = {
            "development": "🟢",
            "staging": "🟡",
            "production": "🔴",
        }.get(settings.app_env, "⚪")
        
        st.caption(f"{env_color} {settings.app_env.upper()}")
        
        # Navigation info
        state = get_state()
        if state.page_history:
            st.caption(f"📚 Pages visited: {len(state.page_history)}")
        
        # Quick actions
        st.markdown("### Quick Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Clear Cache", use_container_width=True):
                from app.caching import clear_all_caches
                clear_all_caches()
                st.success("Cache cleared!")
                st.rerun()
        
        with col2:
            if st.button("📊 Show Stats", use_container_width=True):
                from app.caching import get_cache_stats
                stats = get_cache_stats()
                st.json(stats)
        
        # User preferences
        with st.expander("⚙️ Preferences"):
            show_code = st.checkbox(
                "Show code snippets",
                value=state.get_preference("show_code", True),
                key="pref_show_code",
            )
            state.update_preference("show_code", show_code)
            
            auto_run = st.checkbox(
                "Auto-run examples",
                value=state.get_preference("auto_run", False),
                key="pref_auto_run",
            )
            state.update_preference("auto_run", auto_run)
        
        # Debug info (only in development)
        if settings.debug:
            with st.expander("🐛 Debug Info"):
                st.json(state.to_dict())
                
                if st.button("Reset State"):
                    state.reset()
                    st.rerun()
        
        st.markdown("---")
        st.markdown(
            """
            **Built with:**  
            • [Streamlit](https://streamlit.io)  
            • [FastAPI](https://fastapi.tiangolo.com)  
            • [Scikit-learn](https://scikit-learn.org)  
            • [Plotly](https://plotly.com)
            """
        )


def setup_main_content() -> None:
    """Set up the main content area."""
    # Welcome message
    st.title(f"Welcome to {settings.app_name}!")
    
    st.markdown("""
    ## 🚀 Explore Data Science Concepts
    
    This platform provides interactive explanations and examples for various
    data science concepts. Each page focuses on a specific topic with:
    
    - **Theory**: Clear explanations of the concept
    - **Mathematics**: Formulations and equations
    - **Interactive Examples**: Run code and see results
    - **Implementation**: Code snippets in multiple libraries
    - **Applications**: Real-world use cases
    - **Pitfalls**: Common mistakes and how to avoid them
    
    ### 📚 Available Concepts
    
    Use the sidebar to navigate between pages. Currently available:
    
    1. **Linear Regression** - Predicting continuous values
    2. **K-Means Clustering** - Grouping similar data points
    
    ### 🛠️ How to Use
    
    1. Select a concept from the sidebar
    2. Read the theory section to understand the concept
    3. Adjust parameters in the interactive example
    4. Click "Run Example" to see the algorithm in action
    5. Explore different implementations in the code tabs
    
    ### 🎯 Learning Goals
    
    - Understand core data science algorithms
    - See how algorithms work with interactive visualizations
    - Learn implementation details in Python
    - Recognize practical applications and limitations
    """)
    
    # Quick start section
    with st.expander("🚀 Quick Start", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 1. Choose Concept")
            st.markdown("Select a data science concept from the sidebar navigation.")
        
        with col2:
            st.markdown("### 2. Explore Theory")
            st.markdown("Read the explanation and mathematical formulation.")
        
        with col3:
            st.markdown("### 3. Run Examples")
            st.markdown("Adjust parameters and run interactive examples.")
    
    # API status (if available)
    try:
        import httpx
        
        with st.expander("🔌 API Status", expanded=False):
            try:
                response = httpx.get(f"http://{settings.api_host}:{settings.api_port}/health", timeout=2)
                if response.status_code == 200:
                    st.success("✅ API is running")
                    health_data = response.json()
                    st.json(health_data)
                else:
                    st.warning(f"⚠️ API returned status {response.status_code}")
            except Exception as e:
                st.info("ℹ️ API server not running. Start it with: `uv run python -m app.api.main`")
    except ImportError:
        pass  # httpx not installed


def main() -> None:
    """Main application entry point."""
    # Setup
    setup_page_config()
    setup_sidebar()
    
    # Track page visit
    state = get_state()
    state.add_to_history("home")
    
    # Main content
    setup_main_content()
    
    # Footer
    st.markdown("---")
    st.caption(
        f"© 2024 {settings.app_name} | "
        f"Version 0.1.0 | "
        f"Environment: {settings.app_env} | "
        f"[Report Issue](https://github.com/yourusername/data-science-platform/issues)"
    )


if __name__ == "__main__":
    main()