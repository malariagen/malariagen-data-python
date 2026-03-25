# Developer setup (Linux)

Welcome! This guide walks you through getting a local development environment up and running on Linux.
If you prefer a video walkthrough, check out [this VS Code version](https://youtu.be/zddl3n1DCFM) or [this older PyCharm version](https://youtu.be/QniQi-Hoo9A).

## 1. Fork and clone this repo

Start by creating your own copy of the repo on GitHub (fork it), then download it to your machine:

```bash
git clone git@github.com:[username]/malariagen-data-python.git
cd malariagen-data-python
```

## 2. Install Python

We recommend Python 3.12, which is the version used in CI. How you install it depends on your distro:

**Ubuntu** — add the Deadsnakes PPA and install Python 3.12:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.12 python3.12-venv
```

**Debian or other Linux distributions** — the Deadsnakes PPA is Ubuntu-only and won't work here.
Use [pyenv](https://github.com/pyenv/pyenv) instead, which compiles Python from source and works on any distro:

```bash
# 1. Install the libraries Python needs to compile
sudo apt install -y build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev curl libncursesw5-dev xz-utils \
    tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# 2. Download and install pyenv
curl https://pyenv.run | bash

# 3. Tell your shell where pyenv lives (add these lines to ~/.bashrc or ~/.zshrc)
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# 4. Reload your shell configuration so the changes take effect
source ~/.bashrc   # or: source ~/.zshrc

# 5. Install Python 3.12 and activate it for this session
pyenv install 3.12
pyenv shell 3.12
```

> **Note:** `pyenv install` compiles Python from source, so it may take a few minutes.

## 3. Install pipx and Poetry

[pipx](https://pipx.pypa.io/) installs Python command-line tools in isolated environments so they don't interfere with your project.
[Poetry](https://python-poetry.org/) manages the project's dependencies and virtual environment.

```bash
python3.12 -m pip install --user pipx
python3.12 -m pipx ensurepath
pipx install poetry
```

> **Tip:** After running `pipx ensurepath`, you may need to open a new terminal for the `pipx` and `poetry` commands to be found.

## 4. Create and activate the development environment

Poetry will create a virtual environment and install all the project's dependencies:

```bash
poetry install
poetry shell
```

Once inside `poetry shell`, you're working inside the project's virtual environment. You can type `exit` to leave it.

## 5. Install pre-commit hooks

Pre-commit hooks run the linter and formatter automatically before each of your commits, catching issues early:

```bash
pipx install pre-commit
pre-commit install
```

To run all the checks manually at any time:
```bash
pre-commit run --all-files
```

## 6. Run tests

Check that everything is working with the fast unit tests (no internet access needed):

```bash
poetry run pytest -v tests/anoph
```

If the tests pass, you're all set! 🎉

## 7. Google Cloud authentication (for legacy tests)

Most development doesn't require this step. Legacy integration tests read real data directly from Google Cloud Storage (GCS), so you'll need to apply for data access first.

1. [Request access to MalariaGEN data on GCS](https://malariagen.github.io/vector-data/vobs/vobs-data-access.html).
2. Once access is granted, install the Google Cloud CLI:
   ```bash
   ./install_gcloud.sh
   ```
3. Authenticate with your Google account:
   ```bash
   ./google-cloud-sdk/bin/gcloud auth application-default login
   ```
4. Run the legacy tests:
   ```bash
   poetry run pytest --ignore=tests/anoph -v tests
   ```

> **Heads up:** Tests will be slow on the first run because data is downloaded from GCS. After that, it's cached locally in `gcs_cache/` so subsequent runs are much faster.
