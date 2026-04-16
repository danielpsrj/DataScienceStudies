"""
State management for the Streamlit application.
Provides a clean interface for managing session state across pages.
"""
from typing import Any, Dict, Optional, Union
import streamlit as st

from app.config import settings


class AppState:
    """
    Application state manager.
    
    This class provides a clean interface for managing state across
    Streamlit pages. It wraps Streamlit's session state with type safety
    and additional functionality.
    """
    
    def __init__(self):
        """Initialize the application state."""
        self._ensure_initialized()
    
    def _ensure_initialized(self) -> None:
        """Ensure the session state is initialized with default values."""
        if "initialized" not in st.session_state:
            st.session_state.initialized = True
            st.session_state.selected_dataset = None
            st.session_state.current_model = None
            st.session_state.page_history = []
            st.session_state.user_preferences = {
                "theme": settings.streamlit_theme,
                "show_code": True,
                "auto_run": False,
            }
            st.session_state.cache = {}
    
    @property
    def selected_dataset(self) -> Optional[str]:
        """Get the currently selected dataset."""
        return st.session_state.get("selected_dataset")
    
    @selected_dataset.setter
    def selected_dataset(self, value: Optional[str]) -> None:
        """Set the currently selected dataset."""
        st.session_state.selected_dataset = value
    
    @property
    def current_model(self) -> Optional[str]:
        """Get the current model type."""
        return st.session_state.get("current_model")
    
    @current_model.setter
    def current_model(self, value: Optional[str]) -> None:
        """Set the current model type."""
        st.session_state.current_model = value
    
    @property
    def page_history(self) -> list[str]:
        """Get the page navigation history."""
        return st.session_state.get("page_history", [])
    
    def add_to_history(self, page_name: str) -> None:
        """Add a page to the navigation history."""
        if page_name not in self.page_history:
            st.session_state.page_history.append(page_name)
            # Keep only last 10 pages
            if len(st.session_state.page_history) > 10:
                st.session_state.page_history.pop(0)
    
    def get_previous_page(self) -> Optional[str]:
        """Get the previous page from history."""
        if len(self.page_history) > 1:
            return self.page_history[-2]
        return None
    
    @property
    def user_preferences(self) -> Dict[str, Any]:
        """Get user preferences."""
        return st.session_state.get("user_preferences", {})
    
    def update_preference(self, key: str, value: Any) -> None:
        """Update a user preference."""
        if "user_preferences" not in st.session_state:
            st.session_state.user_preferences = {}
        st.session_state.user_preferences[key] = value
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a user preference."""
        return self.user_preferences.get(key, default)
    
    def cache_get(self, key: str) -> Optional[Any]:
        """Get a value from the application cache."""
        return st.session_state.cache.get(key) if "cache" in st.session_state else None
    
    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the application cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (not implemented in basic version)
        """
        if "cache" not in st.session_state:
            st.session_state.cache = {}
        st.session_state.cache[key] = value
    
    def cache_clear(self, key: Optional[str] = None) -> None:
        """Clear cache entry or entire cache."""
        if "cache" in st.session_state:
            if key is None:
                st.session_state.cache = {}
            elif key in st.session_state.cache:
                del st.session_state.cache[key]
    
    def reset(self) -> None:
        """Reset the application state (except preferences)."""
        preferences = self.user_preferences.copy()
        st.session_state.clear()
        st.session_state.initialized = True
        st.session_state.user_preferences = preferences
        st.session_state.page_history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary (for debugging)."""
        return {
            "selected_dataset": self.selected_dataset,
            "current_model": self.current_model,
            "page_history": self.page_history,
            "user_preferences": self.user_preferences,
            "cache_keys": list(st.session_state.cache.keys()) if "cache" in st.session_state else [],
        }


# Global state instance
def get_state() -> AppState:
    """
    Get the application state instance.
    
    Returns:
        AppState: The application state manager
    """
    return AppState()


# Convenience functions for common operations
def set_selected_dataset(dataset_name: str) -> None:
    """Set the selected dataset."""
    state = get_state()
    state.selected_dataset = dataset_name


def get_selected_dataset() -> Optional[str]:
    """Get the selected dataset."""
    state = get_state()
    return state.selected_dataset


def set_current_model(model_type: str) -> None:
    """Set the current model type."""
    state = get_state()
    state.current_model = model_type


def get_current_model() -> Optional[str]:
    """Get the current model type."""
    state = get_state()
    return state.current_model


def update_user_preference(key: str, value: Any) -> None:
    """Update a user preference."""
    state = get_state()
    state.update_preference(key, value)


def get_user_preference(key: str, default: Any = None) -> Any:
    """Get a user preference."""
    state = get_state()
    return state.get_preference(key, default)


# Example usage in a Streamlit page:
"""
# Example usage in a page:

from app.state import get_state, set_selected_dataset

# Get state instance
state = get_state()

# Use state
state.selected_dataset = "iris"
dataset = state.selected_dataset

# Or use convenience functions
set_selected_dataset("iris")
dataset = get_selected_dataset()

# Track page navigation
state.add_to_history("regression")
previous_page = state.get_previous_page()
"""