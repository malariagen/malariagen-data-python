# Contributing to malariagen-data-python

Thanks for your interest in contributing! Whether you're fixing a bug, adding a feature, or improving the docs, we're glad to have you here. This guide will help you get your environment set up and walk you through the contribution process.

## About the project

This package provides Python tools for accessing and analyzing genomic data from [MalariaGEN](https://www.malariagen.net/), a global research network studying the genomic epidemiology of malaria and its vectors. It provides access to data on _Anopheles_ mosquito species and _Plasmodium_ malaria parasites, with functionality for variant analysis, haplotype clustering, population genetics, and visualization.

## Setting up your development environment

### Prerequisites

You'll need two tools before you start:

- [pipx](https://pipx.pypa.io/) — installs Python CLI tools in isolated environments
- [git](https://git-scm.com/) — for version control

Both can be installed via your distribution's package manager or [Homebrew](https://brew.sh/) on Mac.

### Initial setup

1. **Fork and clone the repository**

   Fork the repository on GitHub so you have your own copy, then clone it locally:

   ```bash
   git clone git@github.com:[your-username]/malariagen-data-python.git
   cd malariagen-data-python
   ```

2. **Add the upstream remote**

   This lets you pull in future changes from the main project:

   ```bash
   git remote add upstream https://github.com/malariagen/malariagen-data-python.git
   ```

3. **Install Poetry**

   [Poetry](https://python-poetry.org/) manages the project's dependencies and virtual environment:

   ```bash
   pipx install poetry
   ```

4. **Install Python 3.12**

   Python 3.12 is the recommended version — it's what CI uses and what the team develops against:

   ```bash
   poetry python install 3.12
   ```

5. **Install the project and its dependencies**

   ```bash
   poetry env use 3.12
   poetry install --with dev,test,docs
   ```

   This installs the runtime dependencies along with the `dev`, `test`, and `docs`
   [dependency groups](https://python-poetry.org/docs/managing-dependencies/#dependency-groups).
   If you only need to run tests, `poetry install --with test` is sufficient.

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

6. **Install pre-commit hooks**

   Pre-commit hooks run the linter and formatter automatically before every commit, so code quality issues are caught early:

   ```bash
   pipx install pre-commit
   pre-commit install
   ```

## Development workflow

### Creating a new feature or fix

1. **Sync with upstream**

   Before starting, make sure your local `master` is up to date:

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

   Fast unit tests using simulated data (no external data access needed):

   ```bash
   poetry run pytest -v tests --ignore tests/integration
   ```

   To run integration tests that read data from GCS, you'll first need to [request access to MalariaGEN data on GCS](https://malariagen.github.io/vector-data/vobs/vobs-data-access.html).

   Once access has been granted, [install the Google Cloud CLI](https://cloud.google.com/sdk/docs/install). E.g., if on Linux:

   ```bash
   ./install_gcloud.sh
   ```

   Then obtain application-default credentials:

   ```bash
   ./google-cloud-sdk/bin/gcloud auth application-default login
   ```

   Once authenticated, run integration tests:

   ```bash
   poetry run pytest -v tests/integration
   ```

   Tests will run slowly the first time, as data will be read from GCS and cached locally in the `gcs_cache` folder.

5. **Check code quality**

   The pre-commit hooks will run automatically on commit, but you can also run them manually at any time:

   ```bash
   pre-commit run --all-files
   ```

6. **Run typechecking**

   Run static typechecking with mypy:

   ```bash
   poetry run mypy malariagen_data tests --ignore-missing-imports
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

Run dynamic type checking with:

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

Run through this checklist to make sure your PR is ready for review:

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

3. **A good PR description includes:**
   - What problem does this solve?
   - How does it solve it?
   - Relevant issue numbers (e.g., "Fixes #123")
   - What testing you did
   - Any breaking changes or migration notes

### Review process

Once your PR is open, a project maintainer will review it. Here's what to expect:

- PRs require approval from a project maintainer before merging
- CI tests must pass (pytest on Python 3.10, 3.11, and 3.12, with NumPy versions `==2.0.2` and `>=2.0.2,<2.1`)
- Address review feedback by pushing new commits to your branch — no need to open a new PR
- Once approved, a maintainer will merge your PR

## Communication

- **Issues**: Use [GitHub Issues](https://github.com/malariagen/malariagen-data-python/issues) for bug reports and feature requests
- **Discussions**: For questions and general discussion, use [GitHub Discussions](https://github.com/malariagen/malariagen-data-python/discussions)
- **Pull requests**: Use PR comments for code review discussions
- **Email**: For data access questions, contact [support@malariagen.net](mailto:support@malariagen.net)

## Finding something to work on

Not sure where to start? Here are some good entry points:

- Issues labeled [`good first issue`](https://github.com/malariagen/malariagen-data-python/labels/good%20first%20issue) — designed to be approachable for new contributors
- Issues labeled [`help wanted`](https://github.com/malariagen/malariagen-data-python/labels/help%20wanted) — areas where the team would love community help
- Improve documentation or add usage examples
- Increase test coverage

## Questions?

Don't hesitate to ask — we'd rather help you get unstuck than have you spin your wheels:

- Open an issue to ask a question
- Start a discussion on [GitHub Discussions](https://github.com/malariagen/malariagen-data-python/discussions)
- Ask directly in your pull request

We appreciate your contributions and will do our best to help you succeed!

## License

By contributing to this project, you agree that your contributions will be licensed under the [MIT License](LICENSE).
