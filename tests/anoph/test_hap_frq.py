import random

import pandas as pd
import numpy as np
import xarray as xr
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import ag3 as _ag3
from malariagen_data import af1 as _af1
from malariagen_data.anoph.hap_frq import AnophelesHapFrequencyAnalysis
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
    return AnophelesHapFrequencyAnalysis(
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
    return AnophelesHapFrequencyAnalysis(
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


def check_frequency(x):
    loc_nan = np.isnan(x)
    assert np.all(x[~loc_nan] >= 0)
    assert np.all(x[~loc_nan] <= 1)


def check_hap_frequencies(*, api, df, sample_sets, cohorts, min_cohort_size):
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0

    df_samples = api.sample_metadata(sample_sets=sample_sets)
    if "cohort_" + cohorts in df_samples:
        cohort_column = "cohort_" + cohorts
    else:
        cohort_column = cohorts
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    frq_fields = ["frq_" + s for s in cohort_labels] + ["max_af"]
    expected_fields = frq_fields
    assert sorted(df.columns.tolist()) == sorted(expected_fields)


def check_hap_frequencies_advanced(
    *,
    api,
    ds,
):
    assert isinstance(ds, xr.Dataset)
    check_plot_frequencies_time_series(api, ds)
    check_plot_frequencies_time_series_with_taxa(api, ds)
    check_plot_frequencies_time_series_with_areas(api, ds)
    check_plot_frequencies_interactive_map(api, ds)
    assert set(ds.dims) == {"cohorts", "variants"}

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
    x = ds["event_frequency"].values.astype(float)
    check_frequency(x)


@pytest.mark.parametrize(
    "cohorts", ["admin1_year", "admin2_month", "country", "foobar"]
)
@parametrize_with_cases("fixture,api", cases=".")
def test_hap_frequencies_with_str_cohorts(
    fixture,
    api: AnophelesHapFrequencyAnalysis,
    cohorts,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    min_cohort_size = rng.integers(0, 2)
    region = fixture.random_region_str()

    # Set up call params.
    params = dict(
        region=region,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        sample_sets=sample_sets,
    )

    # Test behaviour with bad cohorts param.
    if cohorts == "foobar":
        with pytest.raises(ValueError):
            api.haplotypes_frequencies(**params)
        return

    # Run the function under test.
    df_hap = api.haplotypes_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_hap)

    # Standard checks.
    check_hap_frequencies(
        api=api,
        df=df_hap,
        sample_sets=sample_sets,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
    )


@pytest.mark.parametrize(
    "area_by, period_by",
    [("admin1_iso", "year"), ("admin2_name", "year")],
)
@parametrize_with_cases("fixture,api", cases=".")
def test_hap_frequencies_advanced(
    fixture, api: AnophelesHapFrequencyAnalysis, area_by, period_by
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    min_cohort_size = rng.integers(0, 2)
    region = fixture.random_region_str()

    # Set up call params.
    params_advanced = dict(
        region=region,
        area_by=area_by,
        period_by=period_by,
        min_cohort_size=min_cohort_size,
        sample_sets=sample_sets,
    )

    # Run the other function under test.
    ds_hap = api.haplotypes_frequencies_advanced(**params_advanced)

    # Standard checks.
    check_hap_frequencies_advanced(api=api, ds=ds_hap)
