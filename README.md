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

-   [Af1 API
    docs](https://malariagen.github.io/malariagen-data-python/latest/Af1.html)

-   [Amin1 API
    docs](https://malariagen.github.io/malariagen-data-python/latest/Amin1.html)

-  [Adir1 API
    docs](https://malariagen.github.io/malariagen-data-python/latest/Adir1.html)

-   [Pf8 API
    docs](https://malariagen.github.io/parasite-data/pf8/api.html)

-   [Pf7 API
    docs](https://malariagen.github.io/parasite-data/pf7/api.html)

-   [Pv4 API
    docs](https://malariagen.github.io/parasite-data/pv4/api.html)

## Release notes (change log)

See [GitHub releases](https://github.com/malariagen/malariagen-data-python/releases)
for release notes.

## Developer setup

To get setup for development, see [this video if you prefer VS Code](https://youtu.be/zddl3n1DCFM), or [this older video if you prefer PyCharm](https://youtu.be/QniQi-Hoo9A), and the instructions below.

> **macOS users:** See the [macOS setup section](#developer-setup-macos) below.

Fork and clone this repo:

```bash
git clone git@github.com:[username]/malariagen-data-python.git
```

Install Python, e.g.:

```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.10 python3.10-venv
```

Install pipx, e.g.:

```bash
python3.10 -m pip install --user pipx
python3.10 -m pipx ensurepath
```

Install [poetry](https://python-poetry.org/docs/#installation), e.g.:

```bash
pipx install poetry
```

Create development environment:

```bash
cd malariagen-data-python
poetry use 3.10
poetry install
```

Activate development environment:

```bash
poetry shell
```

Install pre-commit and pre-commit hooks:

```bash
pipx install pre-commit
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

To run legacy tests which read data from GCS, you'll need to [request access to MalariaGEN data on GCS](https://malariagen.github.io/vector-data/vobs/vobs-data-access.html).

Once access has been granted, [install the Google Cloud CLI](https://cloud.google.com/sdk/docs/install). E.g., if on Linux:

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

## Developer setup (macOS)

The instructions above are Linux-focused. If you are on macOS, follow these steps instead.

### Install Miniconda

Download and install Miniconda for macOS from https://docs.conda.io/en/latest/miniconda.html.
Choose the Apple Silicon installer if you have an M1/M2/M3 chip, or the Intel installer otherwise. You can check with:
```bash
uname -m
# arm64 = Apple Silicon, x86_64 = Intel
```

After installation, close and reopen your terminal for conda to be available.

### Create a conda environment

The package requires Python >=3.10 and <3.13. Python 3.13+ is not currently supported.
```bash
conda create -n malariagen python=3.11
conda activate malariagen
```

### Fork, clone and install
```bash
git clone https://github.com/[username]/malariagen-data-python.git
cd malariagen-data-python
pip install -e ".[dev]"
```

### Install pre-commit hooks
```bash
pre-commit install
```

### Run fast unit tests
```bash
pytest -v tests/anoph
```

### Google Cloud authentication

To run legacy tests or access data from GCS, install the Google Cloud CLI:
```bash
brew install google-cloud-sdk
```

Then authenticate:
```bash
gcloud auth application-default login
```

This opens a browser — log in with any Google account. You will also need to [request access to MalariaGEN data on GCS](https://malariagen.github.io/vector-data/vobs/vobs-data-access.html).

### VS Code terminal integration

To use the `code` command from the terminal:

Open VS Code → `Cmd + Shift + P` → type `Shell Command: Install 'code' command in PATH` → press Enter.

## Release process

Create a new GitHub release. That's it. This will automatically
trigger publishing of a new release to PyPI and a new version of
the documentation via GitHub Actions.

The version switcher for the documentation can then be updated by
modifying the `docs/source/_static/switcher.json` file accordingly.

## Citation

If you use the `malariagen_data` package in a publication
or include any of its functions or code in other materials (_e.g._ training resources),
please cite: [doi.org/10.5281/zenodo.11173411](https://doi.org/10.5281/zenodo.11173411)

Some functions may require additional citations to acknowledge specific contributions. These are indicated in the description for each relevant function.

For any questions, please feel free to contact us at: [support@malariagen.net](mailto:support@malariagen.net)


## Sponsorship

This project is currently supported by the following grants:

* [BMGF INV-068808](https://www.gatesfoundation.org/about/committed-grants/2024/04/inv-068808)
* [BMGF INV-062921](https://www.gatesfoundation.org/about/committed-grants/2024/07/inv-062921)

This project was previously supported by the following grants:

* [BMGF INV-001927](https://www.gatesfoundation.org/about/committed-grants/2019/11/inv001927)
