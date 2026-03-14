 Windows Setup Guide

To get setup for development on Windows, see 
[this video if you prefer VS Code](https://youtu.be/zddl3n1DCFM), 
or [this older video if you prefer PyCharm](https://youtu.be/QniQi-Hoo9A), 
and the instructions below.

 1. Fork and clone this repo

Go to https://github.com/malariagen/malariagen-data-python 
and click the Fork button (top right corner).

Then in PowerShell run:
```bash
git clone https://github.com/[username]/malariagen-data-python.git
cd malariagen-data-python
```

 2. Install Python

Download and install Python 3.10 from:
https://www.python.org/downloads/windows/

> ⚠️ Important: During installation, check the box 
> "Add Python to PATH" before clicking Install.

Verify it worked:
```bash
python --version
```

 3. Install pipx and poetry
```bash
python -m pip install --user pipx
python -m pipx ensurepath
pipx install poetry
```

> ⚠️ After running ensurepath, close and reopen 
> PowerShell before continuing.

 4. Create and activate development environment
```bash
poetry install
poetry shell
```

 5. Install pre-commit hooks
```bash
pipx install pre-commit
pre-commit install
```

 6. Add upstream remote and get latest code
```bash
git remote add upstream https://github.com/malariagen/malariagen-data-python
git pull upstream master
```

> ℹ️ Note: On Windows the default branch is called 
> master not main.

 7. Verify everything works
```bash
python -c "import malariagen_data; print('Setup successful!')"
```

 Common Issues on Windows

poetry not found after install
Close and reopen PowerShell, then try again.

git not recognized  
Install Git from https://git-scm.com/download/win 
and restart PowerShell.

python not recognized
Reinstall Python and make sure to check 
"Add Python to PATH" during installation.

fatal: not a git repository
Make sure you are inside the malariagen-data-python 
folder before running any git commands.
Use: cd malariagen-data-python

error: pathspec main did not match
On Windows use master instead of main.
Run: git checkout master
