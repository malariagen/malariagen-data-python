"""Tests for AnophelesIhsAnalysis using simulated data."""

import random

import bokeh.models
import numpy as np
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.ihs import AnophelesIhsAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesIhsAnalysis(
        url=ag3_sim_fixture.url,
        public_url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
        aim_metadata_dtype={
            "aim_species_fraction_arab": "float64",
            "aim_species_fraction_colu": "float64",
            "aim_species_fraction_colu_no2l": "float64",
            "aim_species_gambcolu_arabiensis": object,
            "aim_species_gambiae_coluzzii": object,
            "aim_species": object,
        },
        gff_gene_type="gene",
        gff_gene_name_attribute="Name",
        gff_default_attributes=("ID", "Parent", "Name", "description"),
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_ag3.TAXON_COLORS,
        default_phasing_analysis="gamb_colu_arab",
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesIhsAnalysis(
        url=af1_sim_fixture.url,
        public_url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        results_cache=af1_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_af1.TAXON_COLORS,
        default_phasing_analysis="funestus",
    )


# N.B., here we use pytest_cases to parametrize tests. Each
# function whose name begins with "case_" defines a set of
# inputs to the test functions. See the documentation for
# pytest_cases for more information, e.g.:
#
# https://smarie.github.io/python-pytest-cases/#basic-usage
#
# We use this approach here because we want to use fixtures
# as test parameters, which is otherwise hard to do with
# pytest alone.


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


def check_ihs_gwss(*, api, ihs_params):
    """Core check for ihs_gwss results and plots."""
    # Run main gwss function under test.
    x, ihs = api.ihs_gwss(**ihs_params)

    # Check types and shapes.
    assert isinstance(x, np.ndarray)
    assert isinstance(ihs, np.ndarray)
    assert x.ndim == 1
    assert x.dtype.kind == "f"

    # When window_size is set (default), ihs can be 1D (single percentile)
    # or 2D (multiple percentiles). Either way the leading dimension matches x.
    assert ihs.shape[0] == x.shape[0]

    if len(x) == 0:
        # With very sparse simulated data, all variants may be filtered
        # and there are no windows to plot; skip plotting checks.
        return

    # Check plotting functions.
    fig = api.plot_ihs_gwss_track(**ihs_params, show=False)
    assert isinstance(fig, bokeh.models.Plot)

    fig = api.plot_ihs_gwss(**ihs_params, show=False)
    assert isinstance(fig, bokeh.models.GridPlot)


@parametrize_with_cases("fixture,api", cases=".")
def test_ihs_gwss_with_default_analysis(fixture, api: AnophelesIhsAnalysis):
    # Skip datasets with no phasing analyses (IHS requires haplotype data).
    if not api.phasing_analysis_ids:
        pytest.skip("No phasing analyses available for this dataset.")

    # Set up test parameters using small cohort sizes for simulated data.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    ihs_params = dict(
        contig=random.choice(api.contigs),
        sample_sets=[random.choice(all_sample_sets)],
        window_size=random.randint(20, 100),
        min_cohort_size=1,
        max_cohort_size=None,
        # Disable standardization for simulated data: too few variants to build
        # reliable standardization bins.
        standardize=False,
        filter_min_maf=0.0,
        compute_min_maf=0.0,
    )

    # Run checks.
    check_ihs_gwss(api=api, ihs_params=ihs_params)


@parametrize_with_cases("fixture,api", cases=".")
def test_ihs_gwss_with_specific_analysis(fixture, api: AnophelesIhsAnalysis):
    # Skip datasets with no phasing analyses.
    if not api.phasing_analysis_ids:
        pytest.skip("No phasing analyses available for this dataset.")

    # Test with each available phasing analysis.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    contig = random.choice(api.contigs)

    for analysis in api.phasing_analysis_ids:
        # Check whether any samples are available for this analysis.
        try:
            ds_hap = api.haplotypes(
                sample_sets=all_sample_sets,
                analysis=analysis,
                region=contig,
            )
        except ValueError:
            # No samples available for this analysis on this contig.
            continue

        n_samples = ds_hap.sizes["samples"]
        ihs_params = dict(
            contig=contig,
            analysis=analysis,
            sample_sets=all_sample_sets,
            window_size=random.randint(20, 100),
            min_cohort_size=n_samples,
            max_cohort_size=None,
            # Disable standardization to avoid failures with small simulated
            # datasets where there may be too few variants per bin.
            standardize=False,
            filter_min_maf=0.0,
            compute_min_maf=0.0,
        )

        check_ihs_gwss(api=api, ihs_params=ihs_params)

        # Check that requesting more samples than available raises ValueError.
        with pytest.raises(ValueError):
            api.ihs_gwss(
                contig=contig,
                analysis=analysis,
                sample_sets=all_sample_sets,
                window_size=random.randint(20, 100),
                min_cohort_size=n_samples + 1,
            )


@parametrize_with_cases("fixture,api", cases=".")
def test_ihs_gwss_without_windowing(fixture, api: AnophelesIhsAnalysis):
    """Test per-variant iHS (window_size=0) returns 1-D arrays."""
    # Skip datasets with no phasing analyses.
    if not api.phasing_analysis_ids:
        pytest.skip("No phasing analyses available for this dataset.")

    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    ihs_params = dict(
        contig=random.choice(api.contigs),
        sample_sets=[random.choice(all_sample_sets)],
        # Use window_size=0 for per-variant iHS (falsy value skips windowing).
        window_size=0,
        min_cohort_size=1,
        max_cohort_size=None,
        # Turn off standardization and MAF filtering so that simulated data
        # with few variants still produces results.
        standardize=False,
        filter_min_maf=0.0,
        compute_min_maf=0.0,
    )

    x, ihs = api.ihs_gwss(**ihs_params)

    # Both arrays must be 1-D and have the same length.
    assert isinstance(x, np.ndarray)
    assert isinstance(ihs, np.ndarray)
    assert x.ndim == 1
    assert ihs.ndim == 1
    assert x.shape == ihs.shape
    # Positions are raw integers when no windowing is applied (no mean is taken).
    assert x.dtype.kind in ("i", "u", "f")


@parametrize_with_cases("fixture,api", cases=".")
def test_ihs_gwss_with_sample_query(fixture, api: AnophelesIhsAnalysis):
    """Test that ihs_gwss accepts a sample_query parameter."""
    # Skip datasets with no phasing analyses.
    if not api.phasing_analysis_ids:
        pytest.skip("No phasing analyses available for this dataset.")

    all_sample_sets = api.sample_sets()["sample_set"].to_list()

    # Pick a country that has samples.
    all_countries = (
        api.sample_metadata(sample_sets=all_sample_sets)["country"].unique().tolist()
    )
    country = random.choice(all_countries)
    sample_query = f"country == '{country}'"

    try:
        x, ihs = api.ihs_gwss(
            contig=random.choice(api.contigs),
            sample_sets=all_sample_sets,
            sample_query=sample_query,
            window_size=random.randint(20, 100),
            min_cohort_size=1,
            max_cohort_size=None,
        )
        assert isinstance(x, np.ndarray)
        assert isinstance(ihs, np.ndarray)
        assert x.ndim == 1
    except ValueError:
        # It's OK if there's no haplotype data for the selected country.
        pass


@parametrize_with_cases("fixture,api", cases=".")
def test_ihs_gwss_caching(fixture, api: AnophelesIhsAnalysis):
    """Test that calling ihs_gwss twice returns consistent cached results."""
    # Skip datasets with no phasing analyses.
    if not api.phasing_analysis_ids:
        pytest.skip("No phasing analyses available for this dataset.")

    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    ihs_params = dict(
        contig=random.choice(api.contigs),
        sample_sets=[random.choice(all_sample_sets)],
        window_size=random.randint(20, 100),
        min_cohort_size=1,
        max_cohort_size=None,
        # Disable standardization to avoid failures with small simulated
        # datasets where there may be too few variants per standardization bin.
        standardize=False,
        filter_min_maf=0.0,
        compute_min_maf=0.0,
    )

    # Call twice - second call should use cache.
    x1, ihs1 = api.ihs_gwss(**ihs_params)
    x2, ihs2 = api.ihs_gwss(**ihs_params)

    np.testing.assert_array_equal(x1, x2)
    np.testing.assert_array_equal(ihs1, ihs2)


@parametrize_with_cases("fixture,api", cases=".")
def test_ihs_gwss_multiple_percentiles(fixture, api: AnophelesIhsAnalysis):
    """Test ihs_gwss with multiple percentiles returns correctly shaped output."""
    # Skip datasets with no phasing analyses.
    if not api.phasing_analysis_ids:
        pytest.skip("No phasing analyses available for this dataset.")

    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    percentiles = (50, 75, 100)
    ihs_params = dict(
        contig=random.choice(api.contigs),
        sample_sets=[random.choice(all_sample_sets)],
        window_size=random.randint(20, 100),
        percentiles=percentiles,
        min_cohort_size=1,
        max_cohort_size=None,
        # Disable standardization to avoid failures with small simulated
        # datasets where there may be too few variants per standardization bin.
        standardize=False,
        filter_min_maf=0.0,
        compute_min_maf=0.0,
    )

    x, ihs = api.ihs_gwss(**ihs_params)

    assert isinstance(x, np.ndarray)
    assert isinstance(ihs, np.ndarray)
    assert x.ndim == 1
    # With multiple percentiles, ihs should be 2-D: (n_windows, n_percentiles)
    assert ihs.ndim == 2
    assert ihs.shape[0] == x.shape[0]
    assert ihs.shape[1] == len(percentiles)


@parametrize_with_cases("fixture,api", cases=".")
def test_ihs_gwss_single_percentile(fixture, api: AnophelesIhsAnalysis):
    """Test ihs_gwss with a single integer percentile."""
    # Skip datasets with no phasing analyses.
    if not api.phasing_analysis_ids:
        pytest.skip("No phasing analyses available for this dataset.")

    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    ihs_params = dict(
        contig=random.choice(api.contigs),
        sample_sets=[random.choice(all_sample_sets)],
        window_size=random.randint(20, 100),
        percentiles=50,  # single integer
        min_cohort_size=1,
        max_cohort_size=None,
        # Disable standardization to avoid failures with small simulated data.
        standardize=False,
        filter_min_maf=0.0,
        compute_min_maf=0.0,
    )

    x, ihs = api.ihs_gwss(**ihs_params)

    assert isinstance(x, np.ndarray)
    assert isinstance(ihs, np.ndarray)
    assert x.ndim == 1
    # Single percentile: ihs is 1-D
    assert ihs.ndim == 1
    assert ihs.shape == x.shape
