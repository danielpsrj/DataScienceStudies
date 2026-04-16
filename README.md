# Data Science Platform

A comprehensive Streamlit-based platform for exploring data science concepts with interactive examples, machine learning algorithms, and experiment tracking.

## 🚀 Features

### Core Platform
- **Interactive Concept Exploration**: Learn data science concepts through interactive examples
- **Multi-page Streamlit Application**: Clean, modular architecture with reusable components
- **Experiment Tracking**: Save and track machine learning experiments with metrics
- **Database Integration**: SQLite database with SQLAlchemy ORM for data persistence
- **REST API**: FastAPI backend for programmatic access to platform features
- **Comprehensive Testing**: Pytest test suite with fixtures and integration tests

### Data Science Concepts
- **Regression Analysis**: Linear regression with interactive parameter tuning
- **Clustering Algorithms**: K-means, DBSCAN, Hierarchical, and Gaussian Mixture Models
- **Interactive Visualizations**: Plotly charts with real-time updates
- **Algorithm Comparison**: Compare different algorithms on the same dataset

### Technical Features
- **Modular Architecture**: Clean separation of concerns with reusable components
- **State Management**: Efficient caching and session state management
- **Configuration Management**: Environment-based configuration with Pydantic
- **Database Models**: Comprehensive data models for users, experiments, datasets, and concepts
- **Repository Pattern**: Clean data access layer with type hints
- **Docker Support**: Containerized deployment for both Streamlit and FastAPI

## 📁 Project Structure

```
data-science-platform/
├── app/                          # Main application
│   ├── __init__.py
│   ├── main.py                  # Streamlit main application
│   ├── config.py                # Configuration management
│   ├── caching.py               # Caching utilities
│   ├── state.py                 # State management
│   ├── api/                     # FastAPI backend
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI application
│   │   └── routes/             # API routes
│   ├── components/              # Reusable Streamlit components
│   │   ├── __init__.py
│   │   ├── applications.py     # Real-world applications
│   │   ├── code.py            # Code examples
│   │   ├── demo.py            # Interactive demos
│   │   ├── math.py            # Mathematical formulations
│   │   ├── pitfalls.py        # Common pitfalls
│   │   └── references.py      # References and resources
│   ├── data/                   # Data layer
│   │   ├── __init__.py
│   │   ├── models/            # SQLAlchemy models
│   │   └── repositories/      # Repository pattern implementation
│   ├── logic/                  # Business logic
│   │   ├── __init__.py
│   │   ├── regression.py      # Regression algorithms
│   │   └── clustering.py      # Clustering algorithms
│   └── pages/                  # Streamlit pages
│       ├── __init__.py
│       ├── 01_📈_regression.py  # Regression concept page
│       └── 02_🔍_clustering.py  # Clustering concept page
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── conftest.py            # Pytest fixtures
│   ├── test_basic.py          # Basic tests
│   ├── test_database.py       # Database tests
│   ├── test_api/              # API tests
│   ├── test_logic/            # Logic tests
│   └── test_pages/            # Page tests
├── scripts/                    # Utility scripts
│   └── generate_fake_data.py  # Generate sample data
├── docker/                     # Docker configuration
│   └── Dockerfile.streamlit   # Streamlit Dockerfile
├── docs/                       # Documentation
│   └── architecture.md        # Architecture documentation
├── pyproject.toml             # Project dependencies and configuration
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- UV package manager (recommended) or pip

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd data-science-platform
   ```

2. **Install dependencies with UV**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env if needed
   ```

4. **Generate sample data**
   ```bash
   python scripts/generate_fake_data.py
   ```

5. **Run the application**
   ```bash
   streamlit run app/main.py
   ```

6. **Run the API server (optional)**
   ```bash
   uvicorn app.api.main:app --reload
   ```

## 🐳 Docker Deployment

### Docker Compose (Recommended)
The platform includes Docker Compose configurations for both production and development environments.

#### Production Deployment
```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

#### Development Deployment
```bash
# Build and start development services
docker-compose -f docker-compose.dev.yml up -d

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Stop services
docker-compose -f docker-compose.dev.yml down
```

### Individual Docker Images

#### Streamlit Application
```bash
# Build the Docker image
docker build -f docker/Dockerfile.streamlit -t data-science-platform-streamlit .

# Run the container
docker run -p 8501:8501 data-science-platform-streamlit
```

#### FastAPI Backend
```bash
# Build the Docker image
docker build -f docker/Dockerfile.api -t data-science-platform-api .

# Run the container
docker run -p 8000:8000 data-science-platform-api
```

### Services Available
- **Streamlit UI**: http://localhost:8501
- **FastAPI API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Redis (optional)**: localhost:6379

## 📊 Database Schema

The platform uses SQLite with the following main tables:

- **users**: Platform users and their information
- **experiments**: Machine learning experiments with metrics and parameters
- **concepts**: Data science concepts with descriptions and metadata
- **algorithms**: Machine learning algorithms with strengths and weaknesses
- **datasets**: Dataset information and metadata
- **saved_models**: Serialized machine learning models
- **visualizations**: Visualization data for experiments

## 🧪 Testing

Run the test suite:
```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_database.py -v
pytest tests/test_logic/ -v

# Run with coverage
pytest --cov=app tests/
```

## 🔧 Configuration

Configuration is managed through environment variables and `app/config.py`:

- **Database**: SQLite by default, configurable via `DATABASE_URL`
- **API Settings**: Host, port, and workers configuration
- **Streamlit Settings**: Theme, host, and port
- **Caching**: Redis support for production caching
- **Security**: Secret key and allowed hosts

## 🚀 Usage

### Exploring Concepts
1. Launch the Streamlit application
2. Navigate to different concept pages using the sidebar
3. Interact with parameters and see real-time updates
4. Save interesting experiments to the database

### Using the API
The FastAPI backend provides REST endpoints for:
- Experiment management
- Dataset access
- Model persistence
- User management

API documentation is available at `http://localhost:8000/docs` when the API server is running.

## 📈 Development

### Adding New Concepts
1. Create a new logic module in `app/logic/`
2. Add a corresponding page in `app/pages/`
3. Update the database models if needed
4. Add tests for the new functionality

### Code Style
The project uses:
- **Ruff** for linting and formatting
- **Type hints** throughout the codebase
- **Pydantic** for data validation
- **SQLAlchemy 2.0** for database operations

Run linting:
```bash
ruff check .
ruff format .
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/) for the interactive frontend
- [FastAPI](https://fastapi.tiangolo.com/) for the backend API
- [SQLAlchemy](https://www.sqlalchemy.org/) for database operations
- [Plotly](https://plotly.com/) for interactive visualizations
- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms