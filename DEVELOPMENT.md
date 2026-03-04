# Development Guide

This guide provides comprehensive information for developers contributing to the malariagen-data-python project.

## Quick Start

### Prerequisites
- Python 3.10, 3.11, or 3.12
- [Poetry](https://python-poetry.org/) for dependency management
- [Git](https://git-scm.com/) for version control
- [Pre-commit](https://pre-commit.com/) for code quality

### Setup
```bash
# Clone the repository
git clone https://github.com/malariagen/malariagen-data-python.git
cd malariagen-data-python

# Install dependencies and set up development environment
poetry install

# Install pre-commit hooks
pre-commit install

# Verify your setup
python scripts/verify_setup.py
```

## Development Workflow

### 1. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Your Changes
- Write code following the existing style patterns
- Add tests for new functionality
- Update documentation as needed

### 3. Run Quality Checks
```bash
# Run linting and formatting
ruff check .
ruff format .

# Run tests
poetry run pytest tests/

# Run type checking
poetry run mypy malariagen_data/

# Verify documentation builds
cd docs && poetry run make html
```

### 4. Commit Your Changes
```bash
git add .
git commit -m "feat: add your descriptive message"
```

### 5. Push and Create Pull Request
```bash
git push origin feature/your-feature-name
```

## Code Quality Standards

### Linting and Formatting
We use [Ruff](https://docs.astral.sh/ruff/) for both linting and formatting:
- **Line length**: 88 characters
- **Import sorting**: isort-style
- **Code style**: Follows PEP 8 with additional rules

### Type Checking
We use [MyPy](https://mypy.readthedocs.io/) for static type checking:
- Gradual adoption approach (not strictly enforced yet)
- Focus on new code and critical paths
- External dependencies are ignored

### Testing
- **Unit tests**: Fast tests with simulated data (`tests/anoph/`)
- **Integration tests**: Slower tests requiring data access (`tests/integration/`)
- **Coverage**: Aim to maintain or improve test coverage
- **Markers**: Use `@pytest.mark.slow` for time-intensive tests

## Project Structure

```
malariagen_data/
├── __init__.py              # Main API exports
├── anoph/                   # Anopheles-specific implementations
├── ag3.py, af1.py, etc.     # Species-specific API classes
├── util.py                  # Shared utilities
├── veff.py                  # Variant effect prediction
└── ...

tests/
├── anoph/                   # Unit tests (fast)
├── integration/             # Integration tests (slow)
└── test_*.py               # Other test files

docs/
└── source/                 # Sphinx documentation source

scripts/
└── verify_setup.py        # Development environment verification
```

## Configuration Files

### pyproject.toml
This file contains all project configuration:
- **Dependencies**: Production and development dependencies
- **Ruff**: Linting and formatting rules
- **MyPy**: Type checking configuration
- **Pytest**: Test configuration and markers
- **Coverage**: Test coverage settings

### .pre-commit-config.yaml
Pre-commit hooks that run automatically:
- Trailing whitespace removal
- End-of-file fixing
- YAML validation
- Ruff linting and formatting

## Documentation

### API Documentation
- All public APIs must have comprehensive docstrings
- Follow [numpydoc](https://numpydoc.readthedocs.io/) style guide
- Documentation is built using Sphinx with the pydata theme

### Building Documentation
```bash
cd docs
poetry run make html
# View the output in docs/_build/html/index.html
```

### Adding New APIs
When adding a new API class:
1. Create a new RST file in `docs/source/`
2. Add the API to `docs/source/index.rst`
3. Follow the existing documentation pattern
4. Test the documentation builds successfully

## Performance Considerations

### Memory Usage
- Use Dask arrays for large datasets
- Implement lazy loading where possible
- Consider memory profiling for new features

### Caching
- Implement caching for expensive computations
- Use the existing caching patterns in the codebase
- Clear caches appropriately when data changes

## Common Development Tasks

### Adding a New Species API
1. Create new module (e.g., `new_species.py`)
2. Inherit from appropriate base class (`AnophelesDataResource` or `PlasmodiumDataResource`)
3. Implement species-specific configuration
4. Add to `__init__.py` exports
5. Create documentation
6. Add tests

### Adding New Analysis Functions
1. Add function to appropriate module
2. Include comprehensive docstring
3. Add type hints
4. Write unit tests
5. Update API documentation

### Debugging Common Issues

#### Import Errors
```bash
# Verify installation
poetry run python -c "import malariagen_data"

# Check specific API
poetry run python -c "from malariagen_data import Ag3"
```

#### Test Failures
```bash
# Run specific test file
poetry run pytest tests/test_specific.py -v

# Run with debugging output
poetry run pytest tests/test_specific.py -v -s
```

#### Documentation Build Issues
```bash
# Clean build
cd docs
rm -rf _build/
poetry run make html

# Check for specific warnings
poetry run make html 2>&1 | grep -i warning
```

## Release Process

Releases are automated through GitHub Actions:
1. Create a new GitHub release
2. Version is automatically determined from git tags
3. Package is published to PyPI
4. Documentation is built and deployed

## Getting Help

- **Issues**: Use [GitHub Issues](https://github.com/malariagen/malariagen-data-python/issues) for bug reports
- **Discussions**: Use [GitHub Discussions](https://github.com/malariagen/malariagen-data-python/discussions) for questions
- **Email**: Contact support@malariagen.net for data access questions

## Development Best Practices

### Code Style
- Write clear, descriptive variable names
- Add comments for complex logic
- Keep functions focused and small
- Use type hints for new code

### Testing
- Write tests before fixing bugs (TDD approach)
- Test edge cases and error conditions
- Use descriptive test names
- Mock external dependencies in unit tests

### Documentation
- Update documentation when changing APIs
- Include examples in docstrings
- Use consistent formatting
- Test documentation builds

### Git Hygiene
- Write clear, descriptive commit messages
- Keep commits focused on single changes
- Use conventional commit format when possible
- Clean up commit history before merging

## Performance Profiling

### Memory Profiling
```bash
poetry run python -m memory_profiler your_script.py
```

### Line Profiling
```bash
poetry run python -m line_profiler your_script.py
```

### Time Profiling
```bash
poetry run python -m cProfile -o profile.stats your_script.py
poetry run python -c "
import pstats
p = pstats.Stats('profile.stats')
p.sort_stats('cumulative').print_stats(20)
"
```

This development guide ensures consistency and quality across all contributions to the malariagen-data-python project.
