import numpy as np
import pandas as pd
import xarray as xr
import pytest
from pytest_cases import parametrize_with_cases
from numpy.testing import assert_allclose, assert_array_equal

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.cnv_frq import AnophelesCnvFrequencyAnalysis
from malariagen_data.util import compare_series_like
from .test_frq import (
    check_plot_frequencies_heatmap,
    check_plot_frequencies_time_series,
    check_plot_frequencies_time_series_with_taxa,
    check_plot_frequencies_time_series_with_areas,
    check_plot_frequencies_interactive_map,
)

rng = np.random.default_rng(seed=42)


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesCnvFrequencyAnalysis(
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
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesCnvFrequencyAnalysis(
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


expected_types = ["amp", "del"]


@pytest.mark.parametrize(
    "cohorts", ["admin1_year", "admin2_month", "country", "foobar"]
)
@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_with_str_cohorts(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
    cohorts,
):
    region = rng.choice(api.contigs)
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = rng.choice(all_sample_sets)
    min_cohort_size = int(rng.integers(0, 2))

    # Set up call params.
    params = dict(
        region=region,
        sample_sets=sample_sets,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        max_coverage_variance=None,
        drop_invariant=False,
    )

    # Test behaviour with bad cohorts param.
    if cohorts == "foobar":
        with pytest.raises(ValueError):
            api.gene_cnv_frequencies(**params)
        return

    # Run the function under test.
    df_cnv = api.gene_cnv_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_cnv)

    # Figure out expected cohort labels.
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    if "cohort_" + cohorts in df_samples:
        cohort_column = "cohort_" + cohorts
    else:
        cohort_column = cohorts
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    # Standard checks.
    check_gene_cnv_frequencies(
        api=api,
        df=df_cnv,
        cohort_labels=cohort_labels,
        region=region,
    )


@pytest.mark.parametrize("min_cohort_size", [0, 10, 100])
@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_with_min_cohort_size(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
    min_cohort_size,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = rng.choice(all_sample_sets)
    region = rng.choice(api.contigs)
    cohorts = "admin1_year"

    # Set up call params.
    params = dict(
        region=region,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        sample_sets=sample_sets,
        max_coverage_variance=None,
        drop_invariant=True,
    )

    # Figure out expected cohort labels.
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    if "cohort_" + cohorts in df_samples:
        cohort_column = "cohort_" + cohorts
    else:
        cohort_column = cohorts
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    if len(cohort_labels) == 0:
        # No cohorts, expect error.
        with pytest.raises(ValueError):
            api.gene_cnv_frequencies(**params)
        return

        # Run the function under test.
    df_cnv = api.gene_cnv_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_cnv)

    # Standard checks.
    check_gene_cnv_frequencies(
        api=api,
        df=df_cnv,
        cohort_labels=cohort_labels,
        region=region,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_with_str_cohorts_and_sample_query(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
):
    # Pick test parameters at random.
    sample_sets = None
    min_cohort_size = 0
    region = rng.choice(api.contigs)
    cohorts = rng.choice(["admin1_year", "admin1_month", "admin2_year", "admin2_month"])
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    countries = df_samples["country"].unique()
    country = rng.choice(countries)
    sample_query = f"country == '{country}'"

    # Figure out expected cohort labels.
    df_samples = api.sample_metadata(sample_sets=sample_sets, sample_query=sample_query)
    cohort_column = "cohort_" + cohorts
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    # Set up call params.
    params = dict(
        region=region,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        sample_sets=sample_sets,
        sample_query=sample_query,
        max_coverage_variance=None,
        drop_invariant=True,
    )

    # Run the function under test.
    df_cnv = api.gene_cnv_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_cnv)

    # Standard checks.
    check_gene_cnv_frequencies(
        api=api,
        df=df_cnv,
        cohort_labels=cohort_labels,
        region=region,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_with_str_cohorts_and_sample_query_options(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
):
    # Pick test parameters at random.
    sample_sets = None
    min_cohort_size = 0
    region = rng.choice(api.contigs)
    cohorts = rng.choice(["admin1_year", "admin1_month", "admin2_year", "admin2_month"])
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    countries = df_samples["country"].unique().tolist()
    countries_list = rng.choice(countries, 2, replace=False).tolist()
    sample_query_options = {
        "local_dict": {
            "countries_list": countries_list,
        }
    }
    sample_query = "country in @countries_list"

    # Figure out expected cohort labels.
    df_samples = api.sample_metadata(
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
    )
    cohort_column = "cohort_" + cohorts
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    # Set up call params.
    params = dict(
        region=region,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        max_coverage_variance=None,
        drop_invariant=True,
    )

    # Run the function under test.
    df_cnv = api.gene_cnv_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_cnv)

    # Standard checks.
    check_gene_cnv_frequencies(
        api=api,
        df=df_cnv,
        cohort_labels=cohort_labels,
        region=region,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_with_dict_cohorts(
    fixture, api: AnophelesCnvFrequencyAnalysis
):
    # Pick test parameters at random.
    sample_sets = None  # all sample sets
    min_cohort_size = int(rng.integers(0, 2))
    region = rng.choice(api.contigs)

    # Create cohorts by country.
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    cohort_counts = df_samples["country"].value_counts()
    cohorts = {cohort: f"country == '{cohort}'" for cohort in cohort_counts.index}
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    # Set up call params.
    params = dict(
        region=region,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        sample_sets=sample_sets,
        max_coverage_variance=None,
        drop_invariant=True,
    )

    # Run the function under test.
    df_cnv = api.gene_cnv_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_cnv)

    # Standard checks.
    check_gene_cnv_frequencies(
        api=api,
        df=df_cnv,
        cohort_labels=cohort_labels,
        region=region,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_without_drop_invariant(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = rng.choice(all_sample_sets)
    min_cohort_size = int(rng.integers(0, 2))
    region = rng.choice(api.contigs)
    cohorts = rng.choice(["admin1_year", "admin2_month", "country"])

    # Figure out expected cohort labels.
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    if "cohort_" + cohorts in df_samples:
        cohort_column = "cohort_" + cohorts
    else:
        cohort_column = cohorts
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    # Set up call params.
    params = dict(
        region=region,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        sample_sets=sample_sets,
        max_coverage_variance=None,
    )

    # Run the function under test.
    df_cnv_a = api.gene_cnv_frequencies(drop_invariant=True, **params)
    df_cnv_b = api.gene_cnv_frequencies(drop_invariant=False, **params)

    check_plot_frequencies_heatmap(api, df_cnv_a)
    check_plot_frequencies_heatmap(api, df_cnv_b)

    # Standard checks.
    check_gene_cnv_frequencies(
        api=api,
        df=df_cnv_a,
        cohort_labels=cohort_labels,
        region=region,
    )
    check_gene_cnv_frequencies(
        api=api,
        df=df_cnv_b,
        cohort_labels=cohort_labels,
        region=region,
    )

    # Check specifics.
    assert len(df_cnv_b) >= len(df_cnv_a)


@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_with_bad_region(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = rng.choice(all_sample_sets)
    min_cohort_size = int(rng.integers(0, 2))
    cohorts = rng.choice(["admin1_year", "admin2_month", "country"])

    # Set up call params.
    params = dict(
        region="foobar",
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        sample_sets=sample_sets,
        drop_invariant=True,
    )

    # Run the function under test.
    with pytest.raises(ValueError):
        api.gene_cnv_frequencies(**params)


@pytest.mark.parametrize("max_coverage_variance", [0, 0.4, 1])
@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_with_max_coverage_variance(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
    max_coverage_variance,
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = rng.choice(all_sample_sets)
    cohorts = rng.choice(["admin1_year", "admin2_month", "country"])
    region = rng.choice(api.contigs)

    params = dict(
        region=region,
        cohorts=cohorts,
        min_cohort_size=0,
        sample_sets=sample_sets,
        max_coverage_variance=None,
    )

    if max_coverage_variance >= 0.4:
        # Expect this to find at least one sample per cohort, so go ahead with full
        # checks.
        df_cnv = api.gene_cnv_frequencies(**params)

        check_plot_frequencies_heatmap(api, df_cnv)

        # Figure out expected cohort labels.
        df_samples = api.sample_metadata(sample_sets=sample_sets)
        if "cohort_" + cohorts in df_samples:
            cohort_column = "cohort_" + cohorts
        else:
            cohort_column = cohorts
        cohort_counts = df_samples[cohort_column].value_counts()
        cohort_labels = cohort_counts[cohort_counts >= 0].index.to_list()

        check_gene_cnv_frequencies(
            api=api,
            df=df_cnv,
            cohort_labels=cohort_labels,
            region=region,
        )
    else:
        # Expect this to find no cohorts.
        with pytest.raises(ValueError):
            api.gene_cnv_frequencies(
                region=region,
                cohorts=cohorts,
                sample_sets=all_sample_sets,
                max_coverage_variance=max_coverage_variance,
            )


@pytest.mark.parametrize("area_by", ["country", "admin1_iso", "admin2_name"])
@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_advanced_with_area_by(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
    area_by,
):
    check_gene_cnv_frequencies_advanced(
        api=api,
        area_by=area_by,
    )


@pytest.mark.parametrize("period_by", ["year", "quarter", "month"])
@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_advanced_with_period_by(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
    period_by,
):
    check_gene_cnv_frequencies_advanced(
        api=api,
        period_by=period_by,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_advanced_with_sample_query(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    df_samples = api.sample_metadata(sample_sets=all_sample_sets)
    countries = df_samples["country"].unique()
    country = rng.choice(countries)
    sample_query = f"country == '{country}'"

    check_gene_cnv_frequencies_advanced(
        api=api,
        sample_sets=all_sample_sets,
        sample_query=sample_query,
        min_cohort_size=0,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_advanced_with_sample_query_options(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    df_samples = api.sample_metadata(sample_sets=all_sample_sets)
    countries = df_samples["country"].unique().tolist()
    countries_list = rng.choice(countries, 2, replace=False).tolist()
    sample_query_options = {
        "local_dict": {
            "countries_list": countries_list,
        }
    }
    sample_query = "country in @countries_list"

    check_gene_cnv_frequencies_advanced(
        api=api,
        sample_sets=all_sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        min_cohort_size=0,
    )


@pytest.mark.parametrize("min_cohort_size", [0, 10, 100])
@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_advanced_with_min_cohort_size(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
    min_cohort_size,
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    area_by = "admin1_iso"
    period_by = "year"
    region = rng.choice(api.contigs)

    if min_cohort_size <= 10:
        # Expect this to find at least one cohort, so go ahead with full
        # checks.
        check_gene_cnv_frequencies_advanced(
            api=api,
            region=region,
            sample_sets=all_sample_sets,
            min_cohort_size=min_cohort_size,
            area_by=area_by,
            period_by=period_by,
        )
    else:
        # Expect this to find no cohorts.
        with pytest.raises(ValueError):
            api.gene_cnv_frequencies_advanced(
                region=region,
                sample_sets=all_sample_sets,
                min_cohort_size=min_cohort_size,
                area_by=area_by,
                period_by=period_by,
                max_coverage_variance=None,
            )


@pytest.mark.parametrize("max_coverage_variance", [0, 0.4, 1])
@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_advanced_with_max_coverage_variance(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
    max_coverage_variance,
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    area_by = "admin1_iso"
    period_by = "year"
    region = rng.choice(api.contigs)

    if max_coverage_variance >= 0.4:
        # Expect this to find at least one cohort, so go ahead with full
        # checks.
        check_gene_cnv_frequencies_advanced(
            api=api,
            region=region,
            sample_sets=all_sample_sets,
            max_coverage_variance=max_coverage_variance,
            area_by=area_by,
            period_by=period_by,
        )
    else:
        # Expect this to find no cohorts.
        with pytest.raises(ValueError):
            api.gene_cnv_frequencies_advanced(
                region=region,
                sample_sets=all_sample_sets,
                area_by=area_by,
                period_by=period_by,
                max_coverage_variance=max_coverage_variance,
            )


@pytest.mark.parametrize("nobs_mode", ["called", "fixed"])
@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_advanced_with_nobs_mode(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
    nobs_mode,
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    area_by = "admin1_iso"
    period_by = "year"
    region = rng.choice(api.contigs)

    check_gene_cnv_frequencies_advanced(
        api=api,
        region=region,
        sample_sets=all_sample_sets,
        nobs_mode=nobs_mode,
        area_by=area_by,
        period_by=period_by,
    )


@pytest.mark.parametrize("variant_query_option", ["amp", "del"])
@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_advanced_with_variant_query(
    fixture,
    api: AnophelesCnvFrequencyAnalysis,
    variant_query_option,
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    area_by = "admin1_iso"
    period_by = "year"
    region = rng.choice(api.contigs)
    variant_query = "cnv_type == '{variant_query_option}'"

    check_gene_cnv_frequencies_advanced(
        api=api,
        region=region,
        sample_sets=all_sample_sets,
        variant_query=variant_query,
        area_by=area_by,
        period_by=period_by,
    )


def check_frequency(x):
    loc_nan = np.isnan(x)
    assert np.all(x[~loc_nan] >= 0)
    assert np.all(x[~loc_nan] <= 1)


def check_gene_cnv_frequencies(
    *,
    api,
    df,
    cohort_labels,
    region,
):
    assert isinstance(df, pd.DataFrame)

    # Check columns.
    universal_fields = [
        "gene_strand",
        "gene_description",
        "contig",
        "start",
        "end",
        "windows",
        "label",
    ]

    frq_fields = ["frq_" + s for s in cohort_labels] + ["max_af"]
    expected_fields = universal_fields + frq_fields
    assert sorted(df.columns.tolist()) == sorted(expected_fields)
    assert df.index.names == ["gene_id", "gene_name", "cnv_type"]

    # Check some values.
    df = df.reset_index()
    assert np.all(df["cnv_type"].isin(expected_types))
    for f in frq_fields:
        x = df[f]
        check_frequency(x)


def check_gene_cnv_frequencies_advanced(
    *,
    api: AnophelesCnvFrequencyAnalysis,
    region=None,
    area_by="admin1_iso",
    period_by="year",
    sample_sets=None,
    sample_query=None,
    sample_query_options=None,
    min_cohort_size=None,
    nobs_mode="called",
    variant_query=None,
    max_coverage_variance=None,
):
    # Pick test parameters at random.
    if region is None:
        region = rng.choice(api.contigs)
    if area_by is None:
        area_by = rng.choice(["country", "admin1_iso", "admin2_name"])
    if period_by is None:
        period_by = rng.choice(["year", "quarter", "month"])
    if sample_sets is None:
        all_sample_sets = api.sample_sets()["sample_set"].to_list()
        sample_sets = rng.choice(all_sample_sets)
    if min_cohort_size is None:
        min_cohort_size = int(rng.integers(0, 2))

    # Run function under test.
    ds = api.gene_cnv_frequencies_advanced(
        region=region,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        min_cohort_size=min_cohort_size,
        variant_query=variant_query,
        max_coverage_variance=max_coverage_variance,
    )

    # Check the result.
    assert isinstance(ds, xr.Dataset)
    check_plot_frequencies_time_series(api, ds)
    check_plot_frequencies_time_series_with_taxa(api, ds)
    check_plot_frequencies_time_series_with_areas(api, ds)
    check_plot_frequencies_interactive_map(api, ds)
    assert set(ds.dims) == {"cohorts", "variants"}

    # Check variant variables.
    expected_variant_vars = [
        "variant_cnv_type",
        "variant_contig",
        "variant_end",
        "variant_gene_id",
        "variant_gene_name",
        "variant_gene_strand",
        "variant_label",
        "variant_max_af",
        "variant_start",
        "variant_windows",
    ]
    for v in expected_variant_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants",)

    # Check cohort variables.
    expected_cohort_vars = [
        "cohort_label",
        "cohort_size",
        "cohort_taxon",
        "cohort_area",
        "cohort_period",
        "cohort_period_start",
        "cohort_period_end",
        "cohort_lat_mean",
        "cohort_lat_min",
        "cohort_lat_max",
        "cohort_lon_mean",
        "cohort_lon_min",
        "cohort_lon_max",
    ]
    for v in expected_cohort_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("cohorts",)

    # Check event variables.
    expected_event_vars = [
        "event_count",
        "event_nobs",
        "event_frequency",
        "event_frequency_ci_low",
        "event_frequency_ci_upp",
    ]
    for v in expected_event_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants", "cohorts")

    # Sanity check for frequency values.
    x = ds["event_frequency"].values
    check_frequency(x)

    # Sanity check area values.
    df_samples = api.sample_metadata(
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
    )
    expected_area_values = np.unique(df_samples[area_by].dropna().values)
    area_values = ds["cohort_area"].values
    # N.B., some areas may not end up in final dataset if cohort
    # size is too small, so do a set membership test
    for a in area_values:
        assert a in expected_area_values

    # Sanity checks for period values.
    period_values = ds["cohort_period"].values
    if period_by == "year":
        expected_freqstr = "Y-DEC"
    elif period_by == "month":
        expected_freqstr = "M"
    elif period_by == "quarter":
        expected_freqstr = "Q-DEC"
    else:
        assert False, "not implemented"
    for p in period_values:
        assert isinstance(p, pd.Period)
        assert p.freqstr == expected_freqstr

    # Sanity check cohort sizes.
    cohort_size_values = ds["cohort_size"].values
    for s in cohort_size_values:
        assert s >= min_cohort_size

    if area_by == "admin1_iso" and period_by == "year":
        # Here we test the behaviour of the function when grouping by admin level
        # 1 and year. We can do some more in-depth testing in this case because
        # we can compare results directly against the simpler snp_allele_frequencies()
        # function with the admin1_year cohorts.

        # Check consistency with the basic snp allele frequencies method.
        df_af = api.gene_cnv_frequencies(
            region=region,
            cohorts="admin1_year",
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            min_cohort_size=min_cohort_size,
            max_coverage_variance=max_coverage_variance,
            include_counts=True,
        )
        # Make sure all variables available to check.
        df_af = df_af.reset_index()
        if variant_query is not None:
            df_af = df_af.query(variant_query)

        # Check cohorts are consistent.
        expect_cohort_labels = sorted(
            [c.split("frq_")[1] for c in df_af.columns if c.startswith("frq_")]
        )
        cohort_labels = sorted(ds["cohort_label"].values)
        assert cohort_labels == expect_cohort_labels

        # Check variants are consistent.
        assert ds.sizes["variants"] == len(df_af)
        for v in expected_variant_vars:
            c = v.split("variant_")[1]
            actual = ds[v]
            expect = df_af[c]
            compare_series_like(actual, expect)

        # Check frequencies are consistent.
        for cohort_index, cohort_label in enumerate(ds["cohort_label"].values):
            actual_nobs = ds["event_nobs"].values[:, cohort_index]
            expect_nobs = df_af[f"nobs_{cohort_label}"].values
            assert_array_equal(actual_nobs, expect_nobs)
            actual_count = ds["event_count"].values[:, cohort_index]
            expect_count = df_af[f"count_{cohort_label}"].values
            assert_array_equal(actual_count, expect_count)
            actual_frq = ds["event_frequency"].values[:, cohort_index]
            expect_frq = df_af[f"frq_{cohort_label}"].values
            assert_allclose(actual_frq, expect_frq)
