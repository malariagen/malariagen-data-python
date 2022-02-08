# `malariagen_data` - access MalariaGEN public data from Python

This Python package provides convenience methods for accessing public
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


## Release notes

### 1.0.0

This is a major release which includes new features and makes some
breaking changes to the public API, see below for more information.


#### New features and API changes

* `Ag3`: Added support for genome regions when accessing data
  ([GH14](https://github.com/malariagen/malariagen-data-python/issues/14)). N.B.,
  the `contig` parameter is no longer supported, instead use the
  `region` parameter which can be a contig ID (e.g., "3L"), a contig
  region (e.g., "3L:1000000-2000000"), a gene ID ("AGAP004070"), or a
  list of any of the above. This affects methods including
  `snp_sites()`, `site_filters()`, `snp_genotypes()` and
  `snp_dataset()`. Contributed by [Nace
  Kranjc](https://github.com/nkran).

* `Ag3`: The parameters for specifying which species analysis version
  is used have changed
  ([GH55](https://github.com/malariagen/malariagen-data-python/issues/55)). This
  affects `species_calls()`, `sample_metadata()`,
  `snp_allele_frequencies()` and `gene_cnv_frequencies()`. In most
  cases the default values for these parameters should be appropriate
  and so no changes to your code should be needed.

* `Ag3`: The names of the columns in dataframes containing data
  related to species calling have changed to make it clearer which
  species calling method has been used. This affects dataframes
  returned by `species_calls()` and `sample_metadata()`. See
  [GH93](https://github.com/malariagen/malariagen-data-python/issues/93)
  for further details.

* `Ag3`: The latest cohorts metadata are now automatically loaded and
  joined in with the sample metadata when calling
  `sample_metadata()`. See
  [GH94](https://github.com/malariagen/malariagen-data-python/issues/94)
  for further details.

* `Ag3`: SNP effects are now automatically included in the output
  dataframe from `snp_allele_frequencies()`
  ([GH95](https://github.com/malariagen/malariagen-data-python/issues/95)).

* `Ag3`: Added a new `sample_query` parameter to methods returning
  frequencies to allow for making a sub-selection of samples
  ([GH96](https://github.com/malariagen/malariagen-data-python/issues/96)).

* `Ag3`: Added a new method `aa_allele_frequencies()` to return a
  dataframe of amino acid substitution allele frequencies
  ([GH101](https://github.com/malariagen/malariagen-data-python/issues/101)).

* `Ag3`: Added a new method `plot_frequencies_heatmap()` for creating
  a heatmap plot of allele frequencies
  ([GH102](https://github.com/malariagen/malariagen-data-python/issues/102)).

* `Ag3`: The Google Cloud Storage URL ("gs://vo_agam_release") is now
  the default value when instantiating the `Ag3` class
  ([GH103](https://github.com/malariagen/malariagen-data-python/issues/103)). So
  now you don't need to provide it if you are accessing data from
  GCS. I.e., you can just do:

```
import malariagen_data
ag3 = malariagen_data.Ag3()
```

* `Ag3`: The identifiers used for data releases have been changed to
  use "3.0" instead of "v3", "3.1" instead of "v3.1",
  etc. ([GH104](https://github.com/malariagen/malariagen-data-python/issues/104))

* `Ag3`: All dataframe columns containing allele frequency values are
  now prefixed with "frq_" to allow for easier selection of frequency
  columns
  ([GH116](https://github.com/malariagen/malariagen-data-python/issues/116)).

* `Ag3`: When computing frequencies, automatically drop columns for
  cohorts below the minimum cohort size
  ([GH118](https://github.com/malariagen/malariagen-data-python/issues/118)).


#### Bug fixes, maintenance and documentation

* `Ag3`: Move default values for analysis parameters to constants
  ([GH70](https://github.com/malariagen/malariagen-data-python/issues/70)).

* `Ag3`: Check for manifest.tsv when discovering a release
  ([GH74](https://github.com/malariagen/malariagen-data-python/issues/74)).

* `Ag3`: Decode sample IDs when building `snp_calls()` dataset
  ([GH82](https://github.com/malariagen/malariagen-data-python/issues/82)).

* `Ag3`: Fix `snp_calls()` cannot take multiple releases for
  `sample_set` parameter
  ([GH85](https://github.com/malariagen/malariagen-data-python/issues/85)).

* `Ag3`: Fix `chunks` parameter appears to be ignored
  ([GH86](https://github.com/malariagen/malariagen-data-python/issues/86)).

* Support Python 3.9
  ([GH91](https://github.com/malariagen/malariagen-data-python/issues/91)).

* `Ag3`: Fix pandas performance warnings
  ([GH108](https://github.com/malariagen/malariagen-data-python/issues/108)).

* `Ag3`: Fix compatibility with zarr 2.11.0
  ([GH129](https://github.com/malariagen/malariagen-data-python/issues/129)).


### 0.15.0

* `Ag3`: Update default cohort parameter to latest analysis
  (20211101).


### 0.14.1

* `Amin1`: Bug fix to `snp_calls()` handling of site_mask parameter.


### 0.14.0

* Adds the `Amin1` class providing access to the *Anopheles minimus*
  `Amin1` SNP data release.


### 0.12.1

* `Ag3`: Bug fix to `sample_cohorts()`.


### 0.12.0

* `Ag3`: Update default cohort parameter to latest analysis
  (20210927).

* `Ag3`: Reduce dataframe fragmentation and memory footprint in
  `gene_cnv_frequencies()`.


### 0.11.0

* `Ag3`: Add support for standard cohorts in the functions
  `snp_allele_frequencies()` and `gene_cnv_frequencies()`.


### 0.10.0

* `Ag3`: Add `sample_cohorts()`.


### 0.9.0

* `Ag3`: Add `haplotypes()` and supporting functions
  `open_haplotypes()` and `open_haplotype_sites()`.


### 0.8.0

* `Ag3`: Add site filter columns to dataframes returned by
  `snp_effects()` and `snp_allele_frequencies()`.


### 0.7.0

* `Ag3`: Rename parameter "populations" to "cohorts" to be consistent
  with sgkit terminology.


### 0.6.0

* `Ag3`: Add `gene_cnv()` and `gene_cnv_frequencies()`.

* `Ag3`: Improvements and maintenance to `snp_effects()` and
  `snp_allele_frequencies()`.


### 0.5.0

* `Ag3`: Add `snp_allele_frequencies()`.

* `Ag3`: Add `snp_effects()`.

* `Ag3`: Add `cnv_hmm()`, `cnv_coverage_calls()` and
  `cnv_discordant_read_calls()`.

* Speed up test suite via caching.

* Add configuration for pre-commit hooks.


### 0.4.3

* Performance improvements for faster reading a indexing zarr arrays.


### 0.4.2

* `Ag3`: Bug fix and minor improvements to `snp_calls()`.


### 0.4.1

* `Ag3`: Explore workarounds to xarray memory issues in the
  `snp_calls()` method.


### 0.4.0

* `Ag3`: Make public the `open_genome()`, `open_snp_sites()`,
  `open_site_filters()` and `open_snp_genotypes()` methods.

* `Ag3`: Add the `cross_metadata()` method.

* `Ag3`: Add `site_annotations()` and `open_site_annotations()`
  methods.

* `Ag3`: Add the `snp_calls()` method.

* Improve unit tests.

* Improve memory usage.


### 0.3.1

* Fix compatibility issue in recent fsspec/gcsfs release.


### 0.3.0

First release with basic functionality in the `Ag3` class for
accessing Ag1000G phase 3 data.


## Developer setup

To get setup for development, see [this
video](https://youtu.be/QniQi-Hoo9A) and the instructions below.

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
