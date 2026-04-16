"""
Repository layer for database operations.
Provides a clean interface for data access.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import create_engine, select, update, delete
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from app.config import settings
from app.data.models import (
    Base,
    User,
    Experiment,
    SavedModel,
    Visualization,
    Dataset,
    Concept,
    Algorithm,
)


class Database:
    """Database connection and session management."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or settings.database_url
        self.engine = create_engine(self.database_url, echo=settings.debug)
        self.SessionLocal = sessionmaker(
            autocommit=False, autoflush=False, bind=self.engine
        )

    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        print(f"Database tables created at {self.database_url}")

    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()

    def close_session(self, session: Session) -> None:
        """Close a database session."""
        session.close()


# Create global database instance
db = Database()


class BaseRepository:
    """Base repository with common database operations."""

    def __init__(self, session: Optional[Session] = None):
        self.session = session or db.get_session()
        self._should_close = session is None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._should_close:
            self.session.close()

    def commit(self) -> None:
        """Commit the current transaction."""
        try:
            self.session.commit()
        except SQLAlchemyError as e:
            self.session.rollback()
            raise e

    def rollback(self) -> None:
        """Rollback the current transaction."""
        self.session.rollback()


class UserRepository(BaseRepository):
    """Repository for User operations."""

    def create_user(self, username: str, email: Optional[str] = None) -> User:
        """Create a new user."""
        user = User(username=username, email=email)
        self.session.add(user)
        self.commit()
        return user

    def get_user(self, user_id: int) -> Optional[User]:
        """Get a user by ID."""
        return self.session.get(User, user_id)

    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get a user by username."""
        stmt = select(User).where(User.username == username)
        return self.session.execute(stmt).scalar_one_or_none()

    def update_user_last_login(self, user_id: int) -> Optional[User]:
        """Update user's last login timestamp."""
        user = self.get_user(user_id)
        if user:
            user.last_login = datetime.utcnow()
            self.commit()
        return user

    def list_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        """List users with pagination."""
        stmt = select(User).offset(skip).limit(limit)
        return list(self.session.execute(stmt).scalars().all())


class ExperimentRepository(BaseRepository):
    """Repository for Experiment operations."""

    def create_experiment(
        self,
        name: str,
        concept_type: str,
        algorithm: str,
        parameters: Dict[str, Any],
        metrics: Dict[str, Any],
        dataset_info: Dict[str, Any],
        user_id: Optional[int] = None,
        description: Optional[str] = None,
        is_public: bool = False,
    ) -> Experiment:
        """Create a new experiment."""
        experiment = Experiment(
            user_id=user_id,
            name=name,
            description=description,
            concept_type=concept_type,
            algorithm=algorithm,
            parameters=parameters,
            metrics=metrics,
            dataset_info=dataset_info,
            is_public=is_public,
        )
        self.session.add(experiment)
        self.commit()
        return experiment

    def get_experiment(self, experiment_id: int) -> Optional[Experiment]:
        """Get an experiment by ID."""
        return self.session.get(Experiment, experiment_id)

    def update_experiment_metrics(
        self,
        experiment_id: int,
        metrics: Dict[str, Any],
    ) -> Optional[Experiment]:
        """Update experiment metrics."""
        experiment = self.get_experiment(experiment_id)
        if experiment:
            # Create a new dictionary to ensure SQLAlchemy detects the change
            updated_metrics = dict(experiment.metrics)
            updated_metrics.update(metrics)
            experiment.metrics = updated_metrics
            experiment.updated_at = datetime.utcnow()
            self.commit()
        return experiment

    def list_experiments(
        self,
        user_id: Optional[int] = None,
        concept_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
        public_only: bool = False,
    ) -> List[Experiment]:
        """List experiments with filtering."""
        stmt = select(Experiment)

        if user_id is not None:
            stmt = stmt.where(Experiment.user_id == user_id)

        if concept_type is not None:
            stmt = stmt.where(Experiment.concept_type == concept_type)

        if public_only:
            stmt = stmt.where(Experiment.is_public == True)

        stmt = stmt.order_by(Experiment.created_at.desc()).offset(skip).limit(limit)
        return list(self.session.execute(stmt).scalars().all())

    def delete_experiment(self, experiment_id: int) -> bool:
        """Delete an experiment."""
        experiment = self.get_experiment(experiment_id)
        if experiment:
            self.session.delete(experiment)
            self.commit()
            return True
        return False


class SavedModelRepository(BaseRepository):
    """Repository for SavedModel operations."""

    def save_model(
        self,
        name: str,
        model_type: str,
        algorithm: str,
        model_data: Dict[str, Any],
        metrics: Dict[str, Any],
        user_id: Optional[int] = None,
        experiment_id: Optional[int] = None,
        feature_names: Optional[List[str]] = None,
        target_name: Optional[str] = None,
        is_public: bool = False,
    ) -> SavedModel:
        """Save a machine learning model."""
        model = SavedModel(
            user_id=user_id,
            experiment_id=experiment_id,
            name=name,
            model_type=model_type,
            algorithm=algorithm,
            model_data=model_data,
            feature_names=feature_names,
            target_name=target_name,
            metrics=metrics,
            is_public=is_public,
        )
        self.session.add(model)
        self.commit()
        return model

    def get_model(self, model_id: int) -> Optional[SavedModel]:
        """Get a saved model by ID."""
        return self.session.get(SavedModel, model_id)

    def list_models(
        self,
        user_id: Optional[int] = None,
        model_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
        public_only: bool = False,
    ) -> List[SavedModel]:
        """List saved models with filtering."""
        stmt = select(SavedModel)

        if user_id is not None:
            stmt = stmt.where(SavedModel.user_id == user_id)

        if model_type is not None:
            stmt = stmt.where(SavedModel.model_type == model_type)

        if public_only:
            stmt = stmt.where(SavedModel.is_public == True)

        stmt = stmt.order_by(SavedModel.created_at.desc()).offset(skip).limit(limit)
        return list(self.session.execute(stmt).scalars().all())


class VisualizationRepository(BaseRepository):
    """Repository for Visualization operations."""

    def save_visualization(
        self,
        experiment_id: int,
        name: str,
        visualization_type: str,
        data: Dict[str, Any],
        config: Optional[Dict[str, Any]] = None,
    ) -> Visualization:
        """Save a visualization."""
        visualization = Visualization(
            experiment_id=experiment_id,
            name=name,
            visualization_type=visualization_type,
            data=data,
            config=config or {},
        )
        self.session.add(visualization)
        self.commit()
        return visualization

    def get_visualization(self, visualization_id: int) -> Optional[Visualization]:
        """Get a visualization by ID."""
        return self.session.get(Visualization, visualization_id)

    def list_visualizations(
        self,
        experiment_id: int,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Visualization]:
        """List visualizations for an experiment."""
        stmt = (
            select(Visualization)
            .where(Visualization.experiment_id == experiment_id)
            .order_by(Visualization.created_at.desc())
            .offset(skip)
            .limit(limit)
        )
        return list(self.session.execute(stmt).scalars().all())


class DatasetRepository(BaseRepository):
    """Repository for Dataset operations."""

    def create_dataset(
        self,
        name: str,
        dataset_type: str,
        columns: List[str],
        num_rows: int,
        num_features: int,
        description: Optional[str] = None,
        source: Optional[str] = None,
        target_column: Optional[str] = None,
        is_public: bool = True,
    ) -> Dataset:
        """Create a new dataset record."""
        dataset = Dataset(
            name=name,
            description=description,
            source=source,
            dataset_type=dataset_type,
            columns=columns,
            num_rows=num_rows,
            num_features=num_features,
            target_column=target_column,
            is_public=is_public,
        )
        self.session.add(dataset)
        self.commit()
        return dataset

    def get_dataset(self, dataset_id: int) -> Optional[Dataset]:
        """Get a dataset by ID."""
        return self.session.get(Dataset, dataset_id)

    def get_dataset_by_name(self, name: str) -> Optional[Dataset]:
        """Get a dataset by name."""
        stmt = select(Dataset).where(Dataset.name == name)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_datasets(
        self,
        dataset_type: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
        public_only: bool = True,
    ) -> List[Dataset]:
        """List datasets with filtering."""
        stmt = select(Dataset)

        if dataset_type is not None:
            stmt = stmt.where(Dataset.dataset_type == dataset_type)

        if public_only:
            stmt = stmt.where(Dataset.is_public == True)

        stmt = stmt.order_by(Dataset.created_at.desc()).offset(skip).limit(limit)
        return list(self.session.execute(stmt).scalars().all())


class ConceptRepository(BaseRepository):
    """Repository for Concept operations."""

    def create_concept(
        self,
        name: str,
        display_name: str,
        category: str,
        description: str,
        difficulty_level: str,
        mathematical_formulation: Optional[str] = None,
        common_algorithms: Optional[List[str]] = None,
        use_cases: Optional[List[str]] = None,
        prerequisites: Optional[List[str]] = None,
    ) -> Concept:
        """Create a new concept."""
        concept = Concept(
            name=name,
            display_name=display_name,
            category=category,
            description=description,
            mathematical_formulation=mathematical_formulation,
            common_algorithms=common_algorithms or [],
            use_cases=use_cases or [],
            difficulty_level=difficulty_level,
            prerequisites=prerequisites or [],
        )
        self.session.add(concept)
        self.commit()
        return concept

    def get_concept(self, concept_id: int) -> Optional[Concept]:
        """Get a concept by ID."""
        return self.session.get(Concept, concept_id)

    def get_concept_by_name(self, name: str) -> Optional[Concept]:
        """Get a concept by name."""
        stmt = select(Concept).where(Concept.name == name)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_concepts(
        self,
        category: Optional[str] = None,
        difficulty_level: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Concept]:
        """List concepts with filtering."""
        stmt = select(Concept)

        if category is not None:
            stmt = stmt.where(Concept.category == category)

        if difficulty_level is not None:
            stmt = stmt.where(Concept.difficulty_level == difficulty_level)

        stmt = stmt.order_by(Concept.name).offset(skip).limit(limit)
        return list(self.session.execute(stmt).scalars().all())


class AlgorithmRepository(BaseRepository):
    """Repository for Algorithm operations."""

    def create_algorithm(
        self,
        name: str,
        display_name: str,
        concept_id: int,
        description: str,
        parameters: Dict[str, Any],
        strengths: List[str],
        weaknesses: List[str],
        implementation_libraries: List[str],
        mathematical_formulation: Optional[str] = None,
    ) -> Algorithm:
        """Create a new algorithm."""
        algorithm = Algorithm(
            name=name,
            display_name=display_name,
            concept_id=concept_id,
            description=description,
            mathematical_formulation=mathematical_formulation,
            parameters=parameters,
            strengths=strengths,
            weaknesses=weaknesses,
            implementation_libraries=implementation_libraries,
        )
        self.session.add(algorithm)
        self.commit()
        return algorithm

    def get_algorithm(self, algorithm_id: int) -> Optional[Algorithm]:
        """Get an algorithm by ID."""
        return self.session.get(Algorithm, algorithm_id)

    def get_algorithm_by_name(self, name: str) -> Optional[Algorithm]:
        """Get an algorithm by name."""
        stmt = select(Algorithm).where(Algorithm.name == name)
        return self.session.execute(stmt).scalar_one_or_none()

    def list_algorithms(
        self,
        concept_id: Optional[int] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[Algorithm]:
        """List algorithms with filtering."""
        stmt = select(Algorithm)

        if concept_id is not None:
            stmt = stmt.where(Algorithm.concept_id == concept_id)

        stmt = stmt.order_by(Algorithm.name).offset(skip).limit(limit)
        return list(self.session.execute(stmt).scalars().all())


# Export repositories
__all__ = [
    "db",
    "UserRepository",
    "ExperimentRepository",
    "SavedModelRepository",
    "VisualizationRepository",
    "DatasetRepository",
    "ConceptRepository",
    "AlgorithmRepository",
]
