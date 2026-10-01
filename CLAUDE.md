# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`malariagen_data` is a Python package for accessing and analysing genomic data from [MalariaGEN](https://www.malariagen.net/): whole-genome sequencing data for *Anopheles* mosquitoes (malaria vectors) and *Plasmodium* parasites, served from Google Cloud Storage. It provides SNP/haplotype/CNV data access plus population-genetics analyses (PCA, Fst, selection scans, clustering, etc.) and Plotly/Bokeh visualisations.

## Common commands

Development uses Poetry with Python 3.10-3.12 (3.12 recommended, matches CI).

```bash
poetry install --with dev,test,docs   # full dev environment
poetry install --with test            # just to run tests

# Fast unit tests (simulated data, no network access)
poetry run pytest -v tests --ignore tests/integration

# Run a single test file / test
poetry run pytest -v tests/anoph/test_pca.py
poetry run pytest -v tests/anoph/test_pca.py::test_pca_defaults

# Integration tests (require real GCS access - see below)
poetry run pytest -v tests/integration

# Dynamic type checking (typeguard) alongside pytest
poetry run pytest -v tests --typeguard-packages=malariagen_data,malariagen_data.anoph

# Static type checking
poetry run mypy malariagen_data tests --ignore-missing-imports

# Lint / format (ruff is also run via pre-commit)
ruff check .
ruff format .
pre-commit run --all-files

# Build the docs (needs Graphviz's `dot` binary on PATH - not declared as a
# project dependency since it's a system package, not a Python one)
poetry run sphinx-build -b html docs/source docs/build/html
```

Integration tests read real data from GCS and require [access to MalariaGEN data](https://malariagen.github.io/vector-data/vobs/vobs-data-access.html) plus [gcloud application-default credentials](https://cloud.google.com/docs/authentication/provide-credentials-adc). Data is cached locally in `gcs_cache/` after the first run.

## Architecture

### Two data domains, two different patterns

- **Anopheles (vector) data** - `Ag3`, `Af1`, `As1`, `Amin1`, `Adir1`, `Adar1` (one class per species/project, in `malariagen_data/ag3.py` etc.) - built from a large mixin composition (see below).
- **Plasmodium (parasite) data** - `Pf7`, `Pf8`, `Pv4` (`malariagen_data/pf7.py`, `pf8.py`, `pv4.py`, sharing `malariagen_data/plasmodium.py`) - simpler, mostly self-contained classes, not part of the mixin system described below.

### The Anopheles mixin composition

Each of the six dataset classes (`Ag3`, `Af1`, `As1`, `Amin1`, `Adir1`, `Adar1`) is a thin, dataset-specific subclass of `AnophelesDataResource` (`malariagen_data/anopheles.py`), which is itself composed from ~25 cooperative mixin classes living in `malariagen_data/anoph/` (e.g. `snp_data.py`, `hap_data.py`, `cnv_data.py`, `pca.py`, `fst.py`, `h12.py`, `sample_metadata.py`, `genome_features.py`, `base.py`). All six dataset classes therefore expose the *same* full method set - what actually varies between them is:

1. **Configuration passed to `super().__init__()`** in each dataset class's `__init__` - e.g. `default_site_mask`, `default_phasing_analysis`, `default_coverage_calls_analysis`, `aim_ids`, GCS bucket URLs. A dataset not having phasing/CNV/AIM data for real is reflected by these being `None`/unset, not by the methods being absent - calling e.g. `haplotypes()` on a dataset with no phasing analysis configured raises a clear error rather than the method not existing.
2. Which underlying data actually exists in GCS for that dataset - some methods are wired up in config but not yet backed by real data for every dataset (verify empirically rather than assuming from the class hierarchy alone).

Mixins use Python's cooperative multiple inheritance (`super().__init__(**kwargs)` chains) - the order mixins are listed in `AnophelesDataResource`'s base class list matters for C3 linearization and is not arbitrary; do not reorder it without understanding the MRO implications.

Each mixin has a matching `*_params.py` file in `malariagen_data/anoph/` (e.g. `pca_params.py`, `hap_params.py`, `base_params.py`) holding shared parameter *type annotations and descriptions* reused across methods, so a parameter's docs aren't redefined per-method.

### Docstring generation

Public methods are decorated with `@doc(...)` from the `numpydoc_decorator` package (not hand-written numpydoc docstrings) - `summary=`, `extended_summary=`, `parameters=`, `returns=`, `notes=` kwargs are assembled into the final docstring. Two things about this that are easy to get wrong (found the hard way while working on the Sphinx docs):
- `extended_summary` is rendered as a single reflowed paragraph - rst directives like `.. versionchanged::` placed there get mangled. Put those in `notes` instead, which preserves paragraph/directive structure.
- A `**kwargs` parameter renders as literal `**kwargs` in the generated docstring, which plain docutils (without `sphinx.ext.napoleon`) parses as unterminated bold markup and warns about. `sphinx.ext.napoleon` is required in `docs/source/conf.py` for these docstrings to render warning-free.

Methods are also wrapped with `@_check_types` (`malariagen_data/util.py`) for runtime argument type checking.

### Data access pattern

Cloud data is accessed via `fsspec`/`gcsfs`, typically wrapped in `simplecache::` for local caching (see `results_cache=` / `simplecache=dict(cache_storage=...)` constructor args used throughout `notebooks/*.ipynb`). Expensive computations (PCA, Fst, selection scans, etc.) are additionally cached via a `results_cache` directory when configured.

## Testing architecture

Fast unit tests (`tests/anoph/`) run against **simulated data** generated on the fly by fixtures in `tests/anoph/conftest.py` and `tests/anoph/fixture/` - small datasets that mimic the real GCS layout/format, generated with a reproducible seeded RNG (see `create_rng()`/`GLOBAL_SEED` in `tests/anoph/conftest.py`) so failures can be reproduced. This is what makes the default `pytest` run fast and network-free. `tests/integration/` holds the separate suite that reads real data from GCS.

## Documentation

Built with Sphinx (`docs/source/conf.py`) + `pydata-sphinx-theme`, deployed via GitHub Actions (`.github/workflows/latest_docs.yml` on push to `master`, `tagged_docs.yml` on version tags) - both just run `poetry run sphinx-build`, with no separate system-package install step, so anything requiring a non-Python system dependency (e.g. Graphviz) needs that documented as a manual prerequisite rather than assumed present in CI.

- Per-dataset pages (`docs/source/Ag3.rst` etc.) use `.. autosummary:: :toctree: generated/` to list curated per-method pages - the `generated/` directory is gitignored, build-generated content.
- `docs/source/architecture.rst` and `docs/source/full_inheritance.rst` document the mixin composition visually via `sphinx.ext.inheritance_diagram` (multiple scoped diagrams using `:top-classes:` to keep individual diagrams readable) and hand-authored `sphinx.ext.graphviz` diagrams (`docs/source/_static/diagrams/*.dot`) for per-dataset views showing which inherited mixins are and aren't backed by real data for that dataset.
- `docs/source/_templates/autosummary/class.rst` overrides Sphinx's default template to drop `:inherited-members:` so a mixin's generated page shows only the methods it contributes directly, not everything it inherits.

## Notebooks

`notebooks/` contains worked-example Jupyter notebooks against real GCS data (one per analysis type, e.g. `plot_pca.ipynb`, `plot_fst_gwss.ipynb`, `karyotype.ipynb`) - useful as a source of proven-working parameter values when writing new examples or debugging a method's expected usage.

## Behavioral Rules
- Always write tests before implementing a feature
- Don't install new packages without asking first
- Always check for existing utility functions before writing new ones
- When uncertain, ask — don't guess

## Structure
-`malariagen-data-python/` - Main directory
    - `/docs` — Documentation, as described above
    - `/malariagen_data/anoph` — functionality for analysing each Anopheles dataset
    - `/malariagen_data` — Top level classes for each data type
    - `/notebooks` — notebooks as described above
    - `/tests` — Tests as described above

# Coding Conventions

- Type hints on all function signatures — parameters and return types
- Parameters for each class in `/malariagen_data/anoph` described in separate `params` file
[//]: # (- Pydantic models for all API input/output — never pass raw dicts across boundaries)
[//]: # (- Use `pathlib.Path` instead of `os.path`)
- f-strings for string formatting (no .format() or % formatting)
[//]: # (- Docstrings on public functions using Google style)
