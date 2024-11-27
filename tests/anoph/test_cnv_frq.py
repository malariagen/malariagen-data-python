import random

import numpy as np
import pandas as pd
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.cnv_frq import AnophelesCnvFrequencyAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesCnvFrequencyAnalysis(
        url=ag3_sim_fixture.url,
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
    region = random.choice(api.contigs)
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    min_cohort_size = random.randint(0, 2)

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
    sample_sets = random.choice(all_sample_sets)
    region = random.choice(api.contigs)
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
    df_snp = api.gene_cnv_frequencies(**params)

    # Standard checks.
    check_gene_cnv_frequencies(
        api=api,
        df=df_snp,
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
    region = random.choice(api.contigs)
    cohorts = random.choice(
        ["admin1_year", "admin1_month", "admin2_year", "admin2_month"]
    )
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    countries = df_samples["country"].unique()
    country = random.choice(countries)
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
    df_snp = api.gene_cnv_frequencies(**params)

    # Standard checks.
    check_gene_cnv_frequencies(
        api=api,
        df=df_snp,
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
    region = random.choice(api.contigs)
    cohorts = random.choice(
        ["admin1_year", "admin1_month", "admin2_year", "admin2_month"]
    )
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    countries = df_samples["country"].unique().tolist()
    countries_list = random.sample(countries, 2)
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
    df_snp = api.gene_cnv_frequencies(**params)

    # Standard checks.
    check_gene_cnv_frequencies(
        api=api,
        df=df_snp,
        cohort_labels=cohort_labels,
        region=region,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_gene_cnv_frequencies_with_dict_cohorts(
    fixture, api: AnophelesCnvFrequencyAnalysis
):
    # Pick test parameters at random.
    sample_sets = None  # all sample sets
    min_cohort_size = random.randint(0, 2)
    region = random.choice(api.contigs)

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
    df_snp = api.gene_cnv_frequencies(**params)

    # Standard checks.
    check_gene_cnv_frequencies(
        api=api,
        df=df_snp,
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
    sample_sets = random.choice(all_sample_sets)
    min_cohort_size = random.randint(0, 2)
    region = random.choice(api.contigs)
    cohorts = random.choice(["admin1_year", "admin2_month", "country"])

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
    sample_sets = random.choice(all_sample_sets)
    min_cohort_size = random.randint(0, 2)
    cohorts = random.choice(["admin1_year", "admin2_month", "country"])

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
