# Contributing to malariagen-data-python

Thanks for your interest in contributing to this project! This guide will help you get started.

## About the project

This package provides Python tools for accessing and analyzing genomic data from [MalariaGEN](https://www.malariagen.net/), a global research network studying the genomic epidemiology of malaria and its vectors. It provides access to data on _Anopheles_ mosquito species and _Plasmodium_ malaria parasites, with functionality for variant analysis, haplotype clustering, population genetics, and visualization.

## Setting up your development environment

### Prerequisites

You'll need:

- Python 3.10.x (CI-tested version)
- [Poetry](https://python-poetry.org/) for dependency management
- [Git](https://git-scm.com/) for version control

### Initial setup

1. **Fork and clone the repository**

   Fork the repository on GitHub, then clone your fork:

   ```bash
   git clone git@github.com:[your-username]/malariagen-data-python.git
   cd malariagen-data-python
   ```

2. **Add the upstream remote**

   ```bash
   git remote add upstream https://github.com/malariagen/malariagen-data-python.git
   ```

3. **Install Poetry** (if not already installed)

   ```bash
   pipx install poetry
   ```

4. **Install the project and its dependencies**

   ```bash
   poetry install
   ```

   **Recommended**: Use `poetry run` to run commands inside the virtual environment:

   ```bash
   poetry run pytest
   poetry run python script.py
   ```

   **Optional**: If you prefer an interactive shell session, install the shell plugin first:

   ```bash
   poetry self add poetry-plugin-shell
   ```

   Then activate the environment with:

   ```bash
   poetry shell
   ```

   After activation, commands run directly inside the virtual environment:

   ```bash
   pytest
   python script.py
   ```

5. **Install pre-commit hooks**

   ```bash
   pipx install pre-commit
   pre-commit install
   ```

   Pre-commit hooks will automatically run `ruff` (linter and formatter) on your changes before each commit.

## Development workflow

### Creating a new feature or fix

1. **Sync with upstream**

   ```bash
   git checkout master
   git pull upstream master
   ```

2. **Create a feature branch**

   If an issue does not already exist for your change, [create one](https://github.com/malariagen/malariagen-data-python/issues/new) first. Then create a branch using the convention `GH{issue number}-{short description}`:

   ```bash
   git checkout -b GH123-fix-broken-filter
   # or
   git checkout -b GH456-add-new-analysis
   ```

3. **Make your changes**

   Write your code, add tests, update documentation as needed.

4. **Run tests locally**

   Fast unit tests (no external data access):

   ```bash
   poetry run pytest -v tests/anoph
   ```

   All unit tests (requires setting up credentials for legacy tests):

   ```bash
   poetry run pytest -v tests --ignore tests/integration
   ```

5. **Check code quality**

   The pre-commit hooks will run automatically, but you can also run them manually:

   ```bash
   pre-commit run --all-files
   ```

### Code style

We use `ruff` for both linting and formatting. The configuration is in `pyproject.toml`. Key points:

- Line length: 88 characters (black default)
- Follow PEP 8 conventions
- Use type hints where appropriate
- Write clear docstrings (we use numpydoc format)

The pre-commit hooks will handle most formatting automatically. If you want to run ruff manually:

```bash
ruff check .
ruff format .
```

### Testing

- **Write tests for new functionality**: Add unit tests in the `tests/` directory
- **Test coverage**: Aim to maintain or improve test coverage
- **Fast tests**: Unit tests should use simulated data when possible (see `tests/anoph/`)
- **Integration tests**: Tests requiring GCS data access are slower and run separately

Run type checking with:

```bash
poetry run pytest -v tests --typeguard-packages=malariagen_data,malariagen_data.anoph
```

### Documentation

- Update docstrings if you modify public APIs
- Documentation is built using Sphinx with the pydata theme
- API docs are auto-generated from docstrings
- Follow the [numpydoc](https://numpydoc.readthedocs.io/) style guide

## Submitting your contribution

### Before opening a pull request

- [ ] Tests pass locally
- [ ] Pre-commit hooks pass (or run `pre-commit run --all-files`)
- [ ] Code is well-documented
- [ ] Commit messages are clear and descriptive

### Opening a pull request

1. **Push your branch**

   ```bash
   git push origin your-branch-name
   ```

2. **Create the pull request**
   - Go to the [repository on GitHub](https://github.com/malariagen/malariagen-data-python)
   - Click "Pull requests" → "New pull request"
   - Select your fork and branch
   - Write a clear title and description

3. **Pull request description should include:**
   - What problem does this solve?
   - How does it solve it?
   - Any relevant issue numbers (e.g., "Fixes #123")
   - Testing done
   - Any breaking changes or migration notes

### Review process

- PRs require approval from a project maintainer
- CI tests must pass (pytest on Python 3.10 with NumPy 1.26.4)
- Address review feedback by pushing new commits to your branch
- Once approved, a maintainer will merge your PR

## AI-assisted contributions

We welcome contributions that involve AI tools (like GitHub Copilot, ChatGPT, or similar). If you use AI assistance:

- Review and understand any AI-generated code before submitting
- Ensure the code follows project conventions and passes all tests
- You remain responsible for the quality and correctness of the contribution
- Disclosure of AI usage is optional. Regardless of tools used, contributors remain responsible for the quality and correctness of their submissions.

## Communication

- **Issues**: Use [GitHub Issues](https://github.com/malariagen/malariagen-data-python/issues) for bug reports and feature requests
- **Discussions**: For questions and general discussion, use [GitHub Discussions](https://github.com/malariagen/malariagen-data-python/discussions)
- **Pull requests**: Use PR comments for code review discussions
- **Email**: For data access questions, contact [support@malariagen.net](mailto:support@malariagen.net)

## Finding something to work on

- Look for issues labeled [`good first issue`](https://github.com/malariagen/malariagen-data-python/labels/good%20first%20issue)
- Check for issues labeled [`help wanted`](https://github.com/malariagen/malariagen-data-python/labels/help%20wanted)
- Improve documentation or add examples
- Increase test coverage

## Questions?

If you're unsure about anything, feel free to:

- Open an issue to ask
- Start a discussion on GitHub Discussions
- Ask in your pull request

We appreciate your contributions and will do our best to help you succeed!

## License

By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE).
