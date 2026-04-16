"""
Caching utilities for the Data Science Platform.
Provides enhanced caching beyond Streamlit's built-in cache.
"""

import hashlib
import json
import pickle
import time
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union
import streamlit as st

from app.config import settings


class CacheEntry:
    """A cache entry with metadata."""

    def __init__(self, value: Any, ttl: Optional[int] = None):
        """
        Initialize a cache entry.

        Args:
            value: The cached value
            ttl: Time to live in seconds (None for no expiration)
        """
        self.value = value
        self.created_at = datetime.now()
        self.ttl = ttl
        self.access_count = 0

    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        expiration_time = self.created_at + timedelta(seconds=self.ttl)
        return datetime.now() > expiration_time

    def access(self) -> Any:
        """Access the cache entry and increment access count."""
        self.access_count += 1
        return self.value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "value": self.value,
            "created_at": self.created_at.isoformat(),
            "ttl": self.ttl,
            "access_count": self.access_count,
        }


class EnhancedCache:
    """
    Enhanced caching system with TTL support and statistics.

    This cache can be used alongside Streamlit's built-in caching
    for more control over cache invalidation and statistics.
    """

    def __init__(self, namespace: str = "default"):
        """
        Initialize the cache.

        Args:
            namespace: Cache namespace to avoid key collisions
        """
        self.namespace = namespace
        self._ensure_initialized()

    def _ensure_initialized(self) -> None:
        """Ensure cache is initialized in session state."""
        cache_key = f"enhanced_cache_{self.namespace}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = {}

    def _get_cache(self) -> Dict[str, CacheEntry]:
        """Get the cache dictionary from session state."""
        cache_key = f"enhanced_cache_{self.namespace}"
        return st.session_state[cache_key]

    def _set_cache(self, cache: Dict[str, CacheEntry]) -> None:
        """Set the cache dictionary in session state."""
        cache_key = f"enhanced_cache_{self.namespace}"
        st.session_state[cache_key] = cache

    def generate_key(self, *args, **kwargs) -> str:
        """
        Generate a cache key from function arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            str: MD5 hash of the serialized arguments
        """
        # Sort kwargs for consistent key generation
        sorted_kwargs = sorted(kwargs.items())

        # Create a string representation
        key_data = {
            "args": args,
            "kwargs": sorted_kwargs,
        }

        # Use JSON for serialization (handles basic types)
        key_str = json.dumps(key_data, sort_keys=True, default=str)

        # Generate MD5 hash
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from the cache.

        Args:
            key: Cache key
            default: Default value if key not found or expired

        Returns:
            The cached value or default
        """
        cache = self._get_cache()

        if key not in cache:
            return default

        entry = cache[key]

        # Check if expired
        if entry.is_expired:
            del cache[key]
            self._set_cache(cache)
            return default

        # Return value and update access count
        return entry.access()

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for no expiration)
        """
        cache = self._get_cache()
        cache[key] = CacheEntry(value, ttl)
        self._set_cache(cache)

    def delete(self, key: str) -> bool:
        """
        Delete a value from the cache.

        Args:
            key: Cache key

        Returns:
            bool: True if key was deleted, False if not found
        """
        cache = self._get_cache()

        if key in cache:
            del cache[key]
            self._set_cache(cache)
            return True

        return False

    def clear(self) -> None:
        """Clear all entries from the cache."""
        self._set_cache({})

    def cleanup(self) -> int:
        """
        Remove expired entries from the cache.

        Returns:
            int: Number of entries removed
        """
        cache = self._get_cache()
        initial_count = len(cache)

        # Remove expired entries
        cache = {k: v for k, v in cache.items() if not v.is_expired}

        self._set_cache(cache)
        return initial_count - len(cache)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        cache = self._get_cache()

        total_entries = len(cache)
        expired_entries = sum(1 for entry in cache.values() if entry.is_expired)
        total_accesses = sum(entry.access_count for entry in cache.values())

        # Calculate average TTL
        ttls = [entry.ttl for entry in cache.values() if entry.ttl is not None]
        avg_ttl = sum(ttls) / len(ttls) if ttls else None

        return {
            "namespace": self.namespace,
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "total_accesses": total_accesses,
            "average_ttl": avg_ttl,
            "size_bytes": len(pickle.dumps(cache)) if cache else 0,
        }

    def cache_function(
        self,
        ttl: Optional[int] = None,
        key_prefix: str = "",
        ignore_args: Optional[list] = None,
        ignore_kwargs: Optional[list] = None,
    ) -> Callable:
        """
        Decorator to cache function results.

        Args:
            ttl: Time to live in seconds
            key_prefix: Prefix for cache keys
            ignore_args: List of argument indices to ignore in key generation
            ignore_kwargs: List of keyword argument names to ignore in key generation

        Returns:
            Decorated function
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Prepare arguments for key generation
                key_args = list(args)
                key_kwargs = kwargs.copy()

                # Remove ignored arguments
                if ignore_args:
                    for idx in sorted(ignore_args, reverse=True):
                        if idx < len(key_args):
                            key_args.pop(idx)

                if ignore_kwargs:
                    for kwarg in ignore_kwargs:
                        key_kwargs.pop(kwarg, None)

                # Generate cache key
                cache_key = self.generate_key(*key_args, **key_kwargs)
                full_key = f"{key_prefix}{func.__name__}_{cache_key}"

                # Try to get from cache
                cached_result = self.get(full_key)
                if cached_result is not None:
                    return cached_result

                # Execute function
                result = func(*args, **kwargs)

                # Store in cache
                self.set(full_key, result, ttl)

                return result

            return wrapper

        return decorator


# Global cache instances
_data_cache = EnhancedCache(namespace="data")
_model_cache = EnhancedCache(namespace="models")
_plot_cache = EnhancedCache(namespace="plots")


# Convenience functions
def cache_data(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """Cache data with optional TTL."""
    _data_cache.set(key, value, ttl)


def get_cached_data(key: str, default: Any = None) -> Any:
    """Get cached data."""
    return _data_cache.get(key, default)


def cache_model(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """Cache model with optional TTL."""
    _model_cache.set(key, value, ttl)


def get_cached_model(key: str, default: Any = None) -> Any:
    """Get cached model."""
    return _model_cache.get(key, default)


def cache_plot(key: str, value: Any, ttl: Optional[int] = None) -> None:
    """Cache plot with optional TTL."""
    _plot_cache.set(key, value, ttl)


def get_cached_plot(key: str, default: Any = None) -> Any:
    """Get cached plot."""
    return _plot_cache.get(key, default)


def clear_all_caches() -> None:
    """Clear all caches."""
    _data_cache.clear()
    _model_cache.clear()
    _plot_cache.clear()


def get_cache_stats() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all caches."""
    return {
        "data": _data_cache.get_stats(),
        "models": _model_cache.get_stats(),
        "plots": _plot_cache.get_stats(),
    }


# Example usage:
"""
# Using the cache decorator
cache = EnhancedCache(namespace="regression")

@cache.cache_function(ttl=3600)
def train_linear_regression(X, y, **kwargs):
    # Expensive training operation
    model = LinearRegression(**kwargs)
    model.fit(X, y)
    return model

# Using convenience functions
cache_data("iris_dataset", iris_data, ttl=1800)
data = get_cached_data("iris_dataset")

# Using Streamlit's cache with enhanced features
@st.cache_data(ttl=300)
def load_dataset(name: str):
    # This uses Streamlit's cache
    return pd.read_csv(f"data/{name}.csv")

# The enhanced cache can be used alongside Streamlit's cache
# for more control over cache invalidation
"""
