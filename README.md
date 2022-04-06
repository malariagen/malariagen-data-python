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


### 4.0.0

* `Ag3`: A new `pca()` function has been added for performing
  principal components analysis
  ([GH187](https://github.com/malariagen/malariagen-data-python/issues/187)).

* `Ag3`: Functions `plot_pca_variance()` and `plot_pca_coords()` have
  been added for plotting PCA results
  ([GH197](https://github.com/malariagen/malariagen-data-python/issues/197)).

* `Ag3`: A new `snp_allele_counts()` function has been added for
  computing SNP allele counts, which is required for various analyses
  such as PCA (locating segregating variants).

* `Ag3`: A new `view_alignments()` function has been added which
  creates an IGV browser in the notebook and adds a track with the
  sequence read alignments from a given sample
  ([GH202](https://github.com/malariagen/malariagen-data-python/issues/202)). There
  is also an `igv()` function for initialising an IGV browser with
  just reference genome and gene tracks.

* `Ag3`: The way that analysis version parameters like
  `cohorts_analysis`, `species_analysis` and `site_filters_analysis`
  are exposed in the API has been simplified
  ([GH203](https://github.com/malariagen/malariagen-data-python/issues/203)). Now
  these parameters are set when the `Ag3` class is instantiated,
  rather than at each method.

* `Ag3`: A check has been added for the location of the machine from
  which requests are being made, and in particular to raise a warning
  in the case where colab allocates a VM outside the US region, which
  results in poor data retrieval performance
  ([GH201](https://github.com/malariagen/malariagen-data-python/issues/201)).

* `Ag3`: By default, bokeh is now automatically configured to output
  plots to the notebook
  ([GH193](https://github.com/malariagen/malariagen-data-python/issues/193)).


### 3.1.0

* `Ag3`: Limit docstring widths for better wrapping in colab help tabs
  ([GH186](https://github.com/malariagen/malariagen-data-python/issues/186)).

* `Ag3`: Return a copy of cached DataFrames to any subsequent user
  modifications do not affect the cached data
  ([GH184](https://github.com/malariagen/malariagen-data-python/issues/184)).

* `Ag3`: Improving zooming behaviour of bokeh genome plots
  ([GH189](https://github.com/malariagen/malariagen-data-python/issues/189)).

* `Ag3`: Add sample identifiers to CNV HMM heatmap plots
  ([GH191](https://github.com/malariagen/malariagen-data-python/issues/191)).

* `Ag3`: Exclude high coverage variance samples by default in CNV HMM
  heatmap plots
  ([GH178](https://github.com/malariagen/malariagen-data-python/issues/178)).

* `Ag3`: Standardise default width of bokeh genome plots
  ([GH174](https://github.com/malariagen/malariagen-data-python/issues/174)).

* `Ag3`: Consistently capitalise plot labels
  ([GH176](https://github.com/malariagen/malariagen-data-python/issues/176)).

* `Ag3`: Tidy title for CNV HMM heatmap plots when using multiple
  sample sets
  ([GH175](https://github.com/malariagen/malariagen-data-python/issues/175)).

* `Ag3`: Fix a bug in loading of gene CNV frequencies where
  intermediate species samples are missing
  ([GH183](https://github.com/malariagen/malariagen-data-python/issues/183)).


### 3.0.0

* Added a new function `Ag3.plot_cnv_hmm_coverage()` which generates a
  bokeh plot showing normalised coverage and HMM copy number for an
  individual sample.

* Added a new function `Ag3.plot_cnv_hmm_heatmap()` which generates a
  bokeh plot showing the HMM copy number for multiple samples as a
  heatmap.

* Added support for accessing genome regions to the CNV data access
  functions `Ag3.cnv_hmm()`, `Ag3.gene_cnv()`,
  `Ag3.gene_cnv_frequencies()` and `Ag3.cnv_coverage_calls()`
  ([GH113](https://github.com/malariagen/malariagen-data-python/issues/113)). Please
  use the `region` parameter to specify a contig or contig region. The
  previous `contig` parameter is no longer supported.

* Added support for a `region` parameter to the `Ag3.geneset()`
  function.

* Added docstrings for `Ag3.plot_genes()` and `Ag3.plot_transcript()`
  ([GH170](https://github.com/malariagen/malariagen-data-python/issues/170)).

* Set plot width and height automatically in
  `Ag3.plot_frequencies_heatmap()` based on the number of rows and
  columns.


### 2.2.0


* Added a new function `Ag3.plot_genes()` which generates a bokeh plot
  of gene annotations
  ([GH154](https://github.com/malariagen/malariagen-data-python/issues/154)).

* Added a new function `Ag3.plot_transcript()` which generates a bokeh
  plot of a gene model
  ([GH155](https://github.com/malariagen/malariagen-data-python/issues/155)).

* Fixed a bug in the `Ag3.gene_cnv_frequencies()` function
  ([GH166](https://github.com/malariagen/malariagen-data-python/issues/166)).

* CI improvements
  ([GH150](https://github.com/malariagen/malariagen-data-python/issues/150)).


### 2.1.0

* `Ag3`: Add support for giving a list of contigs to the `contig`
  parameter in `gene_cnv()` and `gene_cnv_frequencies()`
  ([GH162](https://github.com/malariagen/malariagen-data-python/issues/162)).

* `Ag3`: Miscellaneous optimisations and documentation fixes
  ([GH153](https://github.com/malariagen/malariagen-data-python/issues/153),
  [GH158](https://github.com/malariagen/malariagen-data-python/issues/158),
  [GH159](https://github.com/malariagen/malariagen-data-python/issues/159),
  [GH161](https://github.com/malariagen/malariagen-data-python/issues/161)).


### 2.0.0


#### New features and API changes

* `Ag3`: New functions have been added for space-time analysis of SNP
  allele frequencies and gene CNV frequencies
  ([GH143](https://github.com/malariagen/malariagen-data-python/issues/143)).

  * The new function `plot_frequencies_time_series()` creates faceted time
    series plots of frequencies using plotly.

  * The new function `plot_frequencies_interactive_map()` creates an
    ipyleaflet map with coloured markers representing frequencies in
    different cohorts, with widgets to select the variant, taxon and
    time period of interest.

  * The new function `plot_frequencies_map_markers()` supports plotting
    frequency markers on an existing ipyleaflet map.

  * The new function `snp_allele_frequencies_advanced()` computes SNP
    allele frequencies in a transcript of interest and returns an
    xarray dataset which can be used as input to space and time
    plotting functions.

  * The new function `aa_allele_frequencies_advanced()` computes amino
    acid substitution frequencies in a transcript of interest and
    returns an xarray dataset which can be used as input to space and
    time plotting functions.

  * The new function `gene_cnv_frequences_advanced()` computes gene
    CNV frequencies for a given contig and returns an xarray dataset
    which can be used as input to space and time plotting functions.

  * The function `aa_allele_frequencies()` has been modified
    to better handle the case where SNPs at different genome positions
    cause the same amino acid change.

* `Ag3`: The function `gene_cnv_frequencies()` has been modified so
  that each row now represents a gene and variant (amplification or
  deletion), and columns are cohorts
  ([GH139](https://github.com/malariagen/malariagen-data-python/issues/139)). Also
  a new parameter `drop_invariant` has been added, which is True by
  default, meaning that only records with some evidence of copy number
  variation in the given cohorts are returned.

* `Ag3`: Samples with high coverage variance are now removed by
  default when running the `gene_cnv_frequencies()`, and this can be
  controlled via a new `max_coverage_variance` parameter
  ([GH141](https://github.com/malariagen/malariagen-data-python/issues/141)). To
  support this, the `sample_coverage_variance` variable has been added
  to the output of the `gene_cnv()` function
  ([GH128](https://github.com/malariagen/malariagen-data-python/issues/128)).

* `Ag3`: All functions accepting a `sample_sets` parameter now check
  for the same sample set being selected more than once
  ([GH144](https://github.com/malariagen/malariagen-data-python/issues/144)).

* `Ag3`: The functions which plot frequencies, including
  `plot_frequencies_heatmap()`, `plot_frequencies_time_series()`, and
  `plot_frequencies_interactive_map()`, have been modified to use
  consistent labels for variants
  ([GH145](https://github.com/malariagen/malariagen-data-python/issues/145)).

* `Ag3`: The frequencies plotting functions now automatically set a
  title based on metadata from the input dataframe or dataset
  ([GH146](https://github.com/malariagen/malariagen-data-python/issues/146)). The
  cohorts axis labels have also been moved to the bottom to make room
  for a title.

* `Ag3`: All column names in sample metadata dataframes are now lower
  case, and columns starting "adm" have been renamed to start with
  "admin" (e.g., "adm1_ISO" has been renamed to "admin1_iso") to have
  consistent naming of columns and parameter values relating to
  administrative units
  ([GH142](https://github.com/malariagen/malariagen-data-python/issues/142)).

* `Ag3`: Functions `cnv_hmm()`, `cnv_coverage_calls()` and
  `cnv_discordant_read_calls()` support multiple contigs for the
  `contig` parameter and automatically concatenate datasets
  ([GH90](https://github.com/malariagen/malariagen-data-python/issues/90)).


#### Bug fixes, maintenance and documentation

* `Ag3`: Function docstrings have been improved to document return
  values
  ([GH84](https://github.com/malariagen/malariagen-data-python/issues/84)).

* `Ag3`: Improve repr methods
  ([GH138](https://github.com/malariagen/malariagen-data-python/issues/138)).


### 1.0.1

* `Ag3`: Expose more plotting parameters through the
  `plot_frequencies_heatmap()` method
  ([GH133](https://github.com/malariagen/malariagen-data-python/issues/133)).


### 1.0.0


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

* The `Ag3` and `Amin1` classes have a better repr
  ([GH111](https://github.com/malariagen/malariagen-data-python/issues/111)).

* `Ag3`: All dataframe columns containing allele frequency values are
  now prefixed with "frq_" to allow for easier selection of frequency
  columns
  ([GH116](https://github.com/malariagen/malariagen-data-python/issues/116)).

* `Ag3`: When computing frequencies, automatically drop columns for
  cohorts below the minimum cohort size
  ([GH118](https://github.com/malariagen/malariagen-data-python/issues/118)).

* `Amin1`: Added support for `region` parameter instead of `contig`
  ([GH119](https://github.com/malariagen/malariagen-data-python/issues/119)).

* `Ag3`: The `snp_sites()` method no longer returns a tuple of arrays
  if the `field` parameter is not provided, please provide an explicit
  `field` parameter or use the `snp_calls()` method instead
  (recommended).


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

* `Ag3`: Fix bug involving inconsistent array lengths before and after
  computation
  ([GH114](https://github.com/malariagen/malariagen-data-python/issues/114)).

* `Ag3`: Fix compatibility with zarr 2.11.0
  ([GH129](https://github.com/malariagen/malariagen-data-python/issues/129)).

* Some optimisations to speed up the test suite a bit
  ([GH122](https://github.com/malariagen/malariagen-data-python/issues/122)).


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
