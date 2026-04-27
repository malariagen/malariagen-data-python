# Developer setup (Linux)

## 1. Install Git

Choose the command for your Linux distribution:

**Ubuntu, Debian, and Mint:**

```bash
sudo apt update
sudo apt install -y git
```

**Fedora:**

```bash
sudo dnf install -y git
```

**Arch Linux:**

```bash
sudo pacman -S sudo
sudo pacman -S git
sudo pacman -S openssh
```

If your Arch install does not have `sudo` configured yet, run the commands above as `root`, then configure `sudo` for your user.

## 2. Fork and clone this repo

After forking the repository on GitHub, clone your fork.

Use SSH if your SSH keys are set up:

```bash
git clone git@github.com:[YOUR_GITHUB_USERNAME]/malariagen-data-python.git
cd malariagen-data-python
```

Use HTTPS if you prefer, or if you do not have SSH keys configured (common on WSL):

```bash
git clone https://github.com/[YOUR_GITHUB_USERNAME]/malariagen-data-python.git
cd malariagen-data-python
```

## 3. Install pipx

Choose the command for your Linux distribution:

**Ubuntu, Debian, and Mint:**

```bash
sudo apt update
sudo apt install -y pipx
pipx ensurepath
```

**Fedora:**

```bash
sudo dnf install -y pipx
pipx ensurepath
```

**Arch Linux:**

```bash
sudo pacman -S python-pipx
pipx ensurepath
```

Close and reopen your terminal to apply PATH changes.
If you prefer to reload the shell in-place, run:

```bash
exec bash
```

## 4. Install Poetry and Python 3.12

The package requires `>=3.10,<3.13`. We use Poetry's built-in installer to handle the Python version universally across all distributions.

```bash
pipx install poetry
poetry python install 3.12
```

## 5. Create development environment

```bash
poetry env use 3.12
poetry install --extras dev
```

## 6. Install pre-commit hooks

```bash
pipx install pre-commit
pre-commit install
```

Run pre-commit checks manually:

```bash
pre-commit run --all-files
```

## 7. Run tests

Run fast unit tests using simulated data:

```bash
poetry run pytest -v tests/anoph
```

## 8. Google Cloud authentication (for legacy tests)

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
