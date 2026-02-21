# Developer setup (macOS)

The Linux setup guide is available in [LINUX_SETUP.md](LINUX_SETUP.md). If you are on macOS, follow these steps instead.

## 1. Install Miniconda

Download and install Miniconda for macOS from https://docs.conda.io/en/latest/miniconda.html.
Choose the Apple Silicon installer if you have an Apple Silicon Mac, or the Intel installer otherwise. You can check with:
```bash
uname -m
# arm64 = Apple Silicon, x86_64 = Intel
```

After installation, close and reopen your terminal for conda to be available.

## 2. Create a conda environment

The package requires Python `>=3.10, <3.13`. Python 3.13+ is not currently supported.
```bash
conda create -n malariagen python=3.11
conda activate malariagen
```

## 3. Fork and clone this repo

Fork the repository on GitHub, then clone your fork:
```bash
git clone git@github.com:[username]/malariagen-data-python.git
cd malariagen-data-python
pip install -e ".[dev]"
```

## 4. Install pre-commit hooks
```bash
pre-commit install
```

Run pre-commit checks manually:
```bash
pre-commit run --all-files
```

## 5. Run tests

Run fast unit tests using simulated data:
```bash
pytest -v tests/anoph
```

## 6. Google Cloud authentication (for legacy tests)

To run legacy tests which read data from GCS, you'll need to [request access to MalariaGEN data on GCS](https://malariagen.github.io/vector-data/vobs/vobs-data-access.html).

Once access has been granted, install the Google Cloud CLI:
```bash
brew install google-cloud-sdk
```

Then authenticate:
```bash
gcloud auth application-default login
```

This opens a browser — log in with any Google account.

Once authenticated, run legacy tests:
```bash
pytest --ignore=tests/anoph -v tests
```

Tests will run slowly the first time, as data will be read from GCS and cached locally in the `gcs_cache` folder.

## 7. VS Code terminal integration

To use the `code` command from the terminal:

Open VS Code → `Cmd + Shift + P` → type `Shell Command: Install 'code' command in PATH` → press Enter.