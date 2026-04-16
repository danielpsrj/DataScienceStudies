# AI Development Guide for Data Science Platform

This guide provides comprehensive instructions for AI agents contributing to the Data Science Platform. Follow these guidelines to ensure consistent, high-quality contributions that maintain the project's structure and quality standards.

## 🎯 Quick Start for AI Agents

### 1. **Always Start Here**
```bash
# 1. Navigate to project
cd data-science-platform

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Run tests FIRST
python -m pytest tests/ -v

# 4. Check linting
ruff check .
ruff format .
```

### 2. **Development Workflow**
1. **Tests First**: Always run tests before making changes
2. **Use Templates**: Follow existing patterns and templates
3. **Component-Based**: Use existing components from `app/components/`
4. **Add Tests**: Every new feature needs tests
5. **Run Validation**: Use validation scripts before committing

## 📁 Project Structure Overview

```
data-science-platform/
├── app/
│   ├── components/     # REUSABLE UI COMPONENTS - USE THESE!
│   ├── logic/          # Business logic modules
│   ├── pages/          # Streamlit concept pages
│   ├── data/           # Database models & repositories
│   └── api/            # FastAPI backend
├── tests/              # Test suite
├── scripts/            # Utility scripts
└── template_concept_page.py  # PAGE TEMPLATE - USE THIS!
```

## 🏗️ Creating New Concept Pages

### **MANDATORY: Use the Template**
Always start from `app/pages/template_concept_page.py` when creating new pages. The template enforces the standardized 6-section structure:

### **Required 6-Section Structure**
1. **Concept Overview** (`theory_section` component)
2. **Interactive Demo** (controls in columns, NOT sidebar)
3. **Implementation Examples** (`code_tabs` component)
4. **Real-World Applications** (in interactive tabs)
5. **Common Pitfalls & Fixes** (in interactive tabs)
6. **References & Further Reading** (inside expander)

### **Page Creation Steps**
```python
# 1. Copy template
cp app/pages/template_concept_page.py app/pages/03_📊_new_concept.py

# 2. Update imports
from app.logic.new_concept import (
    generate_data,
    train_model,
    visualize_results,
)

# 3. Implement logic module first
#    Create app/logic/new_concept.py with required functions
```

### **Critical Rules for Pages**
- ✅ **DO**: Keep demo controls in columns within demo section
- ✅ **DO**: Use `app.components` for UI elements
- ✅ **DO**: Present applications/pitfalls in tabs
- ✅ **DO**: Put references in expander
- ❌ **DON'T**: Put demo controls in sidebar
- ❌ **DON'T**: Create custom UI components without checking `app/components/` first
- ❌ **DON'T**: Deviate from the 6-section structure

## 🔧 Using Existing Components

### **Available Components (import from `app.components`)**
- `theory_section()` - Concept overview with image
- `math_equation()` - Mathematical formulations
- `code_tabs()` - Multiple code examples in tabs
- `display_applications()` - Real-world applications
- `display_pitfalls()` - Common pitfalls and fixes
- `display_references()` - References in expander

### **Example Usage**
```python
from app.components import theory_section, math_equation, code_tabs

# Concept overview
theory_section(
    title="Concept Name",
    content="Detailed explanation...",
    image_url="https://example.com/image.png",
    columns=(2, 1),
)

# Mathematical formulation
math_equation(
    equation=r"y = mx + b",
    variables={
        "y": "Dependent variable",
        "m": "Slope",
        "x": "Independent variable",
        "b": "Intercept",
    },
    title="Linear Equation",
    icon="🧮",
    expandable=False,
)

# Code examples
code_tabs({
    "Scikit-learn": "from sklearn import ...",
    "NumPy": "import numpy as np ...",
})
```

## 🧪 Testing Requirements

### **Test Structure**
```
tests/
├── test_logic/
│   └── test_new_concept.py    # Tests for your logic module
├── test_pages/
│   └── test_new_concept_page.py # Tests for your page
└── test_basic.py              # Basic functionality tests
```

### **Required Test Coverage**
- **100%** for new logic modules
- **Key functionality** for pages
- **Edge cases** and error handling
- **Integration tests** for component interactions

### **Test Examples**
```python
# tests/test_logic/test_new_concept.py
def test_generate_data():
    """Test data generation function."""
    data = generate_data(n_samples=100)
    assert data.shape == (100, 2)
    assert isinstance(data, np.ndarray)

def test_train_model():
    """Test model training."""
    X, y = generate_data(n_samples=50)
    model = train_model(X, y)
    assert hasattr(model, 'predict')
    assert hasattr(model, 'score')
```

## 🚀 Development Workflow Checklist

### **Before Starting**
- [ ] Run existing tests: `python -m pytest tests/ -v`
- [ ] Check linting: `ruff check . && ruff format .`
- [ ] Review similar existing pages for patterns

### **During Development**
- [ ] Use `template_concept_page.py` as starting point
- [ ] Implement logic module in `app/logic/` first
- [ ] Create page following 6-section structure
- [ ] Add comprehensive tests
- [ ] Use existing components from `app.components`

### **Before Committing**
- [ ] Run all tests: `python -m pytest tests/ -v`
- [ ] Run validation script: `python scripts/validate_structure.py`
- [ ] Check linting: `ruff check . && ruff format .`
- [ ] Verify page structure matches template
- [ ] Ensure demo controls are NOT in sidebar

## 📝 Code Quality Standards

### **Type Hints (REQUIRED)**
```python
def train_model(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2
) -> Dict[str, Any]:
    """Train model with type hints."""
    # Implementation
    return results
```

### **Docstrings (Google Style)**
```python
def generate_data(n_samples: int, noise: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for demonstration.
    
    Args:
        n_samples: Number of samples to generate
        noise: Standard deviation of Gaussian noise
        
    Returns:
        Tuple of (X, y) where X is features and y is target
        
    Raises:
        ValueError: If n_samples is less than 1
    """
    # Implementation
```

### **Error Handling**
```python
try:
    result = process_data(data)
except ValueError as e:
    st.error(f"Data processing failed: {e}")
    return None
except Exception as e:
    st.error(f"Unexpected error: {e}")
    logger.exception("Unexpected error in process_data")
    return None
```

## 🔍 Validation & Quality Assurance

### **Run Validation Script**
```bash
# Validate page structure
python scripts/validate_page_structure.py app/pages/03_📊_new_concept.py

# Check component usage
python scripts/validate_components.py app/pages/03_📊_new_concept.py
```

### **Common Validation Checks**
1. **Section Count**: Exactly 6 main sections
2. **Demo Controls**: In columns, not sidebar
3. **Component Usage**: Using `app.components` functions
4. **Tab Structure**: Applications and pitfalls in tabs
5. **References**: Inside expander
6. **Imports**: Correct import statements

## 🐛 Troubleshooting Common Issues

### **Tests Failing**
```bash
# Run specific test file
python -m pytest tests/test_logic/test_new_concept.py -v

# Run with debug output
python -m pytest tests/ -v --tb=long

# Check test coverage
python -m pytest tests/ --cov=app.logic.new_concept
```

### **Import Errors**
```bash
# Check if module exists
python -c "import app.logic.new_concept"

# Check import paths
python -c "import sys; print(sys.path)"
```

### **Streamlit Issues**
```bash
# Clear Streamlit cache
rm -rf ~/.streamlit/cache

# Run with debug mode
streamlit run app/main.py --logger.level=debug
```

## 📊 Performance Guidelines

### **Optimization Tips**
- Use `@st.cache_data` for expensive computations
- Limit data size in demos (max 1000 samples)
- Use Plotly for interactive visualizations
- Implement pagination for large data displays

### **Memory Management**
```python
@st.cache_data(ttl=3600)
def expensive_computation(data):
    """Cache expensive computations."""
    return slow_function(data)

# Use generators for large datasets
def process_large_file(file_path):
    with open(file_path) as f:
        for line in f:
            yield process_line(line)
```

## 🤖 AI-Specific Best Practices

### **Pattern Recognition**
- Study `01_📈_regression.py` and `02_🔍_clustering.py` for patterns
- Reuse existing data structures and function signatures
- Follow the same parameter naming conventions

### **Avoid These Anti-Patterns**
- ❌ Creating custom UI components without necessity
- ❌ Putting interactive controls in sidebar
- ❌ Hardcoding values that should be parameters
- ❌ Skipping tests for "simple" features
- ❌ Copy-pasting code without understanding

### **When in Doubt**
1. **Check existing examples** in regression/clustering pages
2. **Use the template** (`template_concept_page.py`)
3. **Run validation scripts** to catch structure issues
4. **Ask for clarification** if patterns are unclear

## 🔄 Update & Maintenance

### **Keeping Current**
- Regularly sync with main branch
- Update dependencies: `uv sync`
- Run full test suite after updates
- Check for deprecated component usage

### **Backward Compatibility**
- Don't break existing pages
- Add new features as optional
- Maintain existing API interfaces
- Update documentation for changes

## 📚 Additional Resources

### **Project Documentation**
- `README.md` - Project overview and setup
- `docs/architecture.md` - System architecture
- `pyproject.toml` - Dependencies and configuration

### **External Resources**
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Python Documentation](https://plotly.com/python/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

## 🎉 Success Checklist

Before considering your contribution complete:

- [ ] All tests pass (`python -m pytest tests/ -v`)
- [ ] No linting errors (`ruff check .`)
- [ ] Page follows 6-section structure
- [ ] Demo controls are in columns (not sidebar)
- [ ] Applications and pitfalls are in tabs
- [ ] References are in expander
- [ ] Comprehensive tests added
- [ ] Documentation updated if needed
- [ ] Validation scripts pass

**Remember**: Consistency is key. Follow existing patterns, use provided components, and maintain the high quality standards of the Data Science Platform.