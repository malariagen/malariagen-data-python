# `malariagen_data` - analyse MalariaGEN data from Python

This Python package provides methods for accessing and analysing data from MalariaGEN.

## Installation

The `malariagen_data` Python package is available from the Python
package index (PyPI) and can be installed via `pip`, e.g.:

```bash
pip install malariagen-data
```

## Documentation

Documentation of classes and methods in the public API are available
from the following locations:

-   [Ag3 API
    docs](https://malariagen.github.io/malariagen-data-python/latest/Ag3.html)

-   [Amin1 API
    docs](https://malariagen.github.io/malariagen-data-python/latest/Amin1.html)

-   [Pf7 API
    docs](https://malariagen.github.io/parasite-data/pf7/api.html)

-   [Pv4 API
    docs](https://malariagen.github.io/parasite-data/pv4/api.html)

## Release notes (change log)

See [GitHub releases](https://github.com/malariagen/malariagen-data-python/releases)
for release notes.

## Developer setup

To get setup for development, see [this
video](https://youtu.be/QniQi-Hoo9A) and the instructions below.

Fork and clone this repo:

```bash
git clone git@github.com:[username]/malariagen-data-python.git
```

Install Python 3.8 (current recommended version for local development), e.g.:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.8 python3.8-venv
```

Install pipx, e.g.:

```bash
python3.8 -m pip install --user pipx
python3.8 -m pipx ensurepath
```

Install [poetry](https://python-poetry.org/docs/#installation), e.g.:

```bash
pipx install poetry==1.4.1 --python=/usr/bin/python3.8
```

Create development environment:

```bash
cd malariagen-data-python
poetry use 3.8
poetry install
```

Activate development environment:

```bash
poetry shell
```

Install pre-commit and pre-commit hooks:

```bash
pipx install pre-commit --python=/usr/bin/python3.8
pre-commit install
```

Run pre-commit checks (isort, black, blackdoc, flake8, ...) manually:

```bash
pre-commit run --all-files
```

Run tests:

```bash
poetry run pytest -v
```

Tests will run slowly the first time, as data required for testing
will be read from GCS. Subsequent runs will be faster as data will be
cached locally in the "gcs_cache" folder.

## Release process

Create a new GitHub release. That's it. This will automatically
trigger publishing of a new release to PyPI and a new version of
the documentation via GitHub Actions.

The version switcher for the documentation can then be updated by
modifying the `docs/source/_static/switcher.json` file accordingly.
