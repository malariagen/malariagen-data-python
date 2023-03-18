# `malariagen_data` - access MalariaGEN data from Python

This Python package provides convenience methods for accessing
data from MalariaGEN.


## Installation

The `malariagen_data` Python package is available from the Python
package index (PyPI) and can be installed via `pip`, e.g.:

```bash
$ pip install malariagen-data
```


## Documentation

Documentation of classes and methods in the public API are available
from the following locations:

* [Ag3 API
  docs](https://malariagen.github.io/vector-data/ag3/api.html)

* [Amin1 API
  docs](https://malariagen.github.io/vector-data/amin1/api.html)

* [Pf7 API
  docs](https://malariagen.github.io/parasite-data/pf7/api.html)

* [Pv4 API
  docs](https://malariagen.github.io/parasite-data/pv4/api.html)


## Release notes (change log)

See [GitHub releases](https://github.com/malariagen/malariagen-data-python/releases)
for release notes.


## Developer setup

To get setup for development, see [this
video](https://youtu.be/QniQi-Hoo9A) and the instructions below.

Fork and clone this repo:

```bash
$ git clone git@github.com:[username]/malariagen-data-python.git
```

Install [poetry](https://python-poetry.org/docs/#installation) 1.3.1 somehow, e.g.:

```bash
$ sudo apt install python3.7-venv
$ python3.7 -m pip install --user pipx
$ python3.7 -m pipx ensurepath
$ pipx install poetry==1.3.1
```

Create development environment:

```bash
$ cd malariagen-data-python
$ poetry install
```

Activate development environment:

```bash
$ poetry shell
```

Install pre-commit hooks:

```bash
$ pre-commit install
```

Run pre-commit checks (isort, black, blackdoc, flake8, ...) manually:

```bash
$ pre-commit run --all-files
```

Run tests:

```bash
$ poetry run pytest -v
```

Tests will run slowly the first time, as data required for testing
will be read from GCS. Subsequent runs will be faster as data will be
cached locally in the "gcs_cache" folder.


## Release process

Create a new GitHub release. That's it. This will automatically
trigger publishing of a new release to PyPI via GitHub actions.
