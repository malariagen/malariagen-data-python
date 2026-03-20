# Developer setup (Linux)

To get setup for development, see [this video if you prefer VS Code](https://youtu.be/zddl3n1DCFM), or [this older video if you prefer PyCharm](https://youtu.be/QniQi-Hoo9A), and the instructions below.

## 1. Fork and clone this repo
```bash
git clone git@github.com:[username]/malariagen-data-python.git
cd malariagen-data-python
```

## 2. Install Python

**Ubuntu** — install from the Deadsnakes PPA:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12 python3.12-venv
```

**Debian / other Linux distributions** — the Deadsnakes PPA is Ubuntu-only. Use [pyenv](https://github.com/pyenv/pyenv) instead:
```bash
# Install pyenv build dependencies (Debian/Ubuntu example)
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev curl libncursesw5-dev xz-utils \
    tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Install pyenv
curl https://pyenv.run | bash

# Add pyenv to your shell (add these lines to ~/.bashrc or ~/.zshrc)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# Reload shell configuration
source ~/.bashrc   # or source ~/.zshrc

# Install and activate Python 3.12
pyenv install 3.12
pyenv shell 3.12
```

## 3. Install pipx and poetry
```bash
python3 -m pip install --user pipx
python3 -m pipx ensurepath
pipx install poetry
```

## 4. Create and activate development environment
```bash
poetry install
poetry shell
```

## 5. Install pre-commit hooks
```bash
pipx install pre-commit
pre-commit install
```

Run pre-commit checks manually:
```bash
pre-commit run --all-files
```

## 6. Run tests

Run fast unit tests using simulated data:
```bash
poetry run pytest -v tests/anoph
```

## 7. Google Cloud authentication (for legacy tests)

To run legacy tests which read data from GCS, you'll need to [request access to MalariaGEN data on GCS](https://malariagen.github.io/vector-data/vobs/vobs-data-access.html).

Once access has been granted, [install the Google Cloud CLI](https://cloud.google.com/sdk/docs/install):
```bash
./install_gcloud.sh
```

Then obtain application-default credentials:
```bash
./google-cloud-sdk/bin/gcloud auth application-default login
```

Once authenticated, run legacy tests:
```bash
poetry run pytest --ignore=tests/anoph -v tests
```

Tests will run slowly the first time, as data will be read from GCS and cached locally in the `gcs_cache` folder.
