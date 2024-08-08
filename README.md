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

Install Python, e.g.:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.9 python3.9-venv
```

Install pipx, e.g.:

```bash
python3.9 -m pip install --user pipx
python3.9 -m pipx ensurepath
```

Install [poetry](https://python-poetry.org/docs/#installation), e.g.:

```bash
pipx install poetry==1.8.2 --python=/usr/bin/python3.9
```

Create development environment:

```bash
cd malariagen-data-python
poetry use 3.9
poetry install
```

Activate development environment:

```bash
poetry shell
```

Install pre-commit and pre-commit hooks:

```bash
pipx install pre-commit --python=/usr/bin/python3.9
pre-commit install
```

Run pre-commit checks (isort, black, blackdoc, flake8, ...) manually:

```bash
pre-commit run --all-files
```

Run fast unit tests using simulated data:

```bash
poetry run pytest -v tests/anoph
```

To run legacy tests which read data from GCS, you'll need to [install the Google Cloud CLI](https://cloud.google.com/sdk/docs/install). E.g., if on Linux:

```bash
./install_gcloud.sh
```

You'll then need to obtain application-default credentials, e.g.:

```bash
./google-cloud-sdk/bin/gcloud auth application-default login
```

Once this is done, you can run legacy tests:

```bash
poetry run pytest --ignore=tests/anoph -v tests
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
