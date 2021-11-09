# `malariagen_data` - access MalariaGEN public data from Python

This Python package provides convenience methods for accessing public data from MalariaGEN.

Installation:

```bash
$ pip install malariagen-data
```

Usage:

```python
import malariagen_data
ag3 = malariagen_data.Ag3("gs://vo_agam_release/")
```

## Developer notes

To get setup for development, see [this video](https://youtu.be/QniQi-Hoo9A) and the instructions below.

Fork and clone this repo:

```bash
$ git clone git@github.com:[username]/malariagen-data-python.git
```

Install [poetry](https://python-poetry.org/docs/#installation) somehow, e.g.:

```bash
$ pip3 install poetry
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
$ pytest -v
```

Bump version, build and publish to PyPI:

```bash
$ poetry version prerelease
$ poetry build
$ poetry publish
```


## Release notes

### 0.14.0

* Adds the `Amin1` class providing access to the *Anopheles minimus* `Amin1` SNP data release.

### 0.12.1

* Bug fix to `Ag3.sample_cohorts()`.

### 0.12.0

* Updates default cohort parameter to latest analysis (20210927).

* Reduces dataframe fragmentation and memory footprint in `Ag3.gene_cnv_frequencies()`.

### 0.11.0

* Adds support for standard cohorts in the functions `Ag3.snp_allele_frequencies()` and `Ag3.gene_cnv_frequencies()`.

### 0.10.0

* Add `Ag3.sample_cohorts()`.

### 0.9.0

* Add `Ag3.haplotypes()` and supporting functions `Ag3.open_haplotypes()`
  and `Ag3.open_haplotype_sites()`.


### 0.8.0

* Add site filter columns to dataframes returned by
  `Ag3.snp_effects()` and `Ag3.snp_allele_frequencies()`.


### 0.7.0

* Rename parameter "populations" to "cohorts" to be consistent with
  sgkit terminology.


### 0.6.0

* Add `Ag3.gene_cnv()` and `Ag3.gene_cnv_frequencies()`.

* Improvements and maintenance to `Ag3.snp_effects()` and
  `Ag3.snp_allele_frequencies()`.


### 0.5.0

* Add `Ag3.snp_allele_frequencies()`.

* Add `Ag3.snp_effects()`.

* Add `Ag3.cnv_hmm()`, `Ag3.cnv_coverage_calls()` and
  `Ag3.cnv_discordant_read_calls()`.

* Speed up test suite via caching.

* Add configuration for pre-commit hooks.


### 0.4.3

* Performance improvements for faster reading a indexing
  zarr arrays.


### 0.4.2

* Bug fix and minor improvements to `Ag3.snp_calls()`.


### 0.4.1

* Explore workarounds to xarray memory issues in the `Ag3.snp_calls()`
  method.


### 0.4.0

* Make public the `Ag3.open_genome()`, `Ag3.open_snp_sites()`,
  `Ag3.open_site_filters()` and `Ag3.open_snp_genotypes()` methods.

* Add the `Ag3.cross_metadata()` method.

* Add `Ag3.site_annotations()` and `Ag3.open_site_annotations()`
  methods.

* Add the `Ag3.snp_calls()` method.

* Improve unit tests.

* Improve memory usage.


### 0.3.1

* Fix compatibility issue in recent fsspec/gcsfs release.


### 0.3.0

First release with basic functionality in the `Ag3` class for
accessing Ag1000G phase 3 data.
