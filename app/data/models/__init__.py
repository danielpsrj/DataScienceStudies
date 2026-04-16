"""
Database models for the Data Science Platform.
"""
from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.ext.declarative import declarative_base


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


class User(Base):
    """User model for tracking platform usage."""
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    email: Mapped[Optional[str]] = mapped_column(String(200), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    last_login: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Relationships
    experiments: Mapped[List["Experiment"]] = relationship("Experiment", back_populates="user", cascade="all, delete-orphan")
    saved_models: Mapped[List["SavedModel"]] = relationship("SavedModel", back_populates="user", cascade="all, delete-orphan")


class Experiment(Base):
    """Experiment model for tracking data science experiments."""
    __tablename__ = "experiments"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    concept_type: Mapped[str] = mapped_column(String(50), index=True)  # regression, clustering, classification, etc.
    algorithm: Mapped[str] = mapped_column(String(100))  # linear_regression, kmeans, etc.
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)  # Algorithm parameters
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)  # Performance metrics
    dataset_info: Mapped[dict] = mapped_column(JSON, default=dict)  # Dataset information
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="experiments")
    visualizations: Mapped[List["Visualization"]] = relationship("Visualization", back_populates="experiment", cascade="all, delete-orphan")


class SavedModel(Base):
    """Saved machine learning model."""
    __tablename__ = "saved_models"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True)
    experiment_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("experiments.id"), nullable=True)
    name: Mapped[str] = mapped_column(String(200), index=True)
    model_type: Mapped[str] = mapped_column(String(50))  # regression, clustering, etc.
    algorithm: Mapped[str] = mapped_column(String(100))  # linear_regression, kmeans, etc.
    model_data: Mapped[dict] = mapped_column(JSON)  # Serialized model parameters
    feature_names: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True)
    target_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="saved_models")
    experiment: Mapped[Optional["Experiment"]] = relationship("Experiment")


class Visualization(Base):
    """Visualization data for experiments."""
    __tablename__ = "visualizations"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    experiment_id: Mapped[int] = mapped_column(Integer, ForeignKey("experiments.id"))
    name: Mapped[str] = mapped_column(String(200))
    visualization_type: Mapped[str] = mapped_column(String(50))  # scatter, line, bar, heatmap, etc.
    data: Mapped[dict] = mapped_column(JSON)  # Plot data
    config: Mapped[dict] = mapped_column(JSON, default=dict)  # Plot configuration
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    experiment: Mapped["Experiment"] = relationship("Experiment", back_populates="visualizations")


class Dataset(Base):
    """Dataset information."""
    __tablename__ = "datasets"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(200), unique=True, index=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)  # URL or file path
    dataset_type: Mapped[str] = mapped_column(String(50))  # synthetic, csv, api, etc.
    columns: Mapped[List[str]] = mapped_column(JSON)  # Column names
    num_rows: Mapped[int] = mapped_column(Integer)
    num_features: Mapped[int] = mapped_column(Integer)
    target_column: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_public: Mapped[bool] = mapped_column(Boolean, default=True)


class Concept(Base):
    """Data science concept information."""
    __tablename__ = "concepts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(50))  # supervised, unsupervised, etc.
    description: Mapped[str] = mapped_column(Text)
    mathematical_formulation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    common_algorithms: Mapped[List[str]] = mapped_column(JSON, default=list)
    use_cases: Mapped[List[str]] = mapped_column(JSON, default=list)
    difficulty_level: Mapped[str] = mapped_column(String(20))  # beginner, intermediate, advanced
    prerequisites: Mapped[List[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Algorithm(Base):
    """Algorithm information."""
    __tablename__ = "algorithms"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(200))
    concept_id: Mapped[int] = mapped_column(Integer, ForeignKey("concepts.id"))
    description: Mapped[str] = mapped_column(Text)
    mathematical_formulation: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    parameters: Mapped[dict] = mapped_column(JSON, default=dict)  # Parameter descriptions and defaults
    strengths: Mapped[List[str]] = mapped_column(JSON, default=list)
    weaknesses: Mapped[List[str]] = mapped_column(JSON, default=list)
    implementation_libraries: Mapped[List[str]] = mapped_column(JSON, default=list)  # sklearn, statsmodels, etc.
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    concept: Mapped["Concept"] = relationship("Concept")


# Export all models
__all__ = [
    "Base",
    "User",
    "Experiment",
    "SavedModel",
    "Visualization",
    "Dataset",
    "Concept",
    "Algorithm",
]