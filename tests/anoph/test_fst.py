import itertools
import random
import pytest
from pytest_cases import parametrize_with_cases
import numpy as np
import bokeh.models
import pandas as pd
import plotly.graph_objects as go

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.fst import AnophelesFstAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesFstAnalysis(
        url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        gcs_url=_ag3.GCS_URL,
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
        default_site_mask="gamb_colu_arab",
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_ag3.TAXON_COLORS,
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesFstAnalysis(
        url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        gcs_url=_af1.GCS_URL,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        default_site_mask="funestus",
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


@parametrize_with_cases("fixture,api", cases=".")
def test_fst_gwss(fixture, api: AnophelesFstAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_countries = api.sample_metadata()["country"].dropna().unique().tolist()
    countries = random.sample(all_countries, 2)
    cohort1_query = f"country == {countries[0]!r}"
    cohort2_query = f"country == {countries[1]!r}"
    fst_params = dict(
        contig=random.choice(api.contigs),
        sample_sets=all_sample_sets,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        site_mask=random.choice(api.site_mask_ids),
        window_size=random.randint(10, 50),
        min_cohort_size=1,
    )

    # Run main gwss function under test.
    x, fst = api.fst_gwss(**fst_params)

    # Check results.
    assert isinstance(x, np.ndarray)
    assert isinstance(fst, np.ndarray)
    assert x.ndim == 1
    assert fst.ndim == 1
    assert x.shape == fst.shape
    assert x.dtype.kind == "f"
    assert fst.dtype.kind == "f"
    assert np.all(fst[~np.isnan(fst)] >= 0)
    assert np.all(fst[~np.isnan(fst)] <= 1)

    # Check plotting functions.
    fig = api.plot_fst_gwss_track(**fst_params, show=False)
    assert isinstance(fig, bokeh.models.Plot)
    fig = api.plot_fst_gwss(**fst_params, show=False)
    assert isinstance(fig, bokeh.models.GridPlot)


@parametrize_with_cases("fixture,api", cases=".")
def test_average_fst(fixture, api: AnophelesFstAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_countries = api.sample_metadata()["country"].dropna().unique().tolist()
    countries = random.sample(all_countries, 2)
    cohort1_query = f"country == {countries[0]!r}"
    cohort2_query = f"country == {countries[1]!r}"
    fst_params = dict(
        region=random.choice(api.contigs),
        sample_sets=all_sample_sets,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        site_mask=random.choice(api.site_mask_ids),
        min_cohort_size=1,
        n_jack=random.randint(10, 200),
    )

    # Run main gwss function under test.
    fst, se = api.average_fst(**fst_params)

    # Checks.
    assert isinstance(fst, float)
    assert isinstance(se, float)
    assert 0 <= fst <= 1
    assert 0 <= se <= 1


@parametrize_with_cases("fixture,api", cases=".")
def test_average_fst_with_min_cohort_size(fixture, api: AnophelesFstAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_countries = api.sample_metadata()["country"].dropna().unique().tolist()
    countries = random.sample(all_countries, 2)
    cohort1_query = f"country == {countries[0]!r}"
    cohort2_query = f"country == {countries[1]!r}"
    fst_params = dict(
        region=random.choice(api.contigs),
        sample_sets=all_sample_sets,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        site_mask=random.choice(api.site_mask_ids),
        min_cohort_size=1000,
    )

    # Run main gwss function under test.
    with pytest.raises(ValueError):
        api.average_fst(**fst_params)


def check_pairwise_average_fst(api: AnophelesFstAnalysis, fst_params):
    # Run main function under test.
    fst_df = api.pairwise_average_fst(**fst_params)

    # Basic checks.
    assert isinstance(fst_df, pd.DataFrame)
    assert fst_df.columns.to_list() == ["cohort1", "cohort2", "fst", "se"]
    assert np.all(fst_df["fst"] >= 0)
    assert np.all(fst_df["fst"] <= 1)
    assert np.all(fst_df["se"] >= 0)
    assert np.all(fst_df["se"] <= 1)

    # Check cohort pairs are correct.
    sample_sets = fst_params["sample_sets"]
    sample_query = fst_params.get("sample_query")
    cohorts = fst_params["cohorts"]
    min_cohort_size = fst_params["min_cohort_size"]
    df_samples = api.sample_metadata(sample_sets=sample_sets, sample_query=sample_query)
    if isinstance(cohorts, str):
        if "cohort_" + cohorts in df_samples:
            cohort_column = "cohort_" + cohorts
        else:
            cohort_column = cohorts
        cohort_counts = df_samples[cohort_column].value_counts()
        expected_cohort_labels = sorted(
            cohort_counts[cohort_counts >= min_cohort_size].index.to_list()
        )
    else:
        assert isinstance(cohorts, dict)
        expected_cohort_labels = list(cohorts.keys())
    expected_pairs = list(itertools.combinations(expected_cohort_labels, 2))
    assert len(fst_df) == len(expected_pairs)
    actual_pairs = list(fst_df[["cohort1", "cohort2"]].itertuples(index=False))
    for expected_pair, actual_pair in zip(expected_pairs, actual_pairs):
        assert expected_pair == actual_pair

    # Check plotting.
    if len(fst_df) > 0:
        fig = api.plot_pairwise_average_fst(fst_df, show=False)
        assert isinstance(fig, go.Figure)
        fig = api.plot_pairwise_average_fst(
            fst_df, annotation="standard error", show=False
        )
        assert isinstance(fig, go.Figure)
        fig = api.plot_pairwise_average_fst(fst_df, annotation="Z score", show=False)
        assert isinstance(fig, go.Figure)


@pytest.mark.parametrize("cohorts", ["country", "admin1_year", "cohort_admin2_month"])
@parametrize_with_cases("fixture,api", cases=".")
def test_pairwise_average_fst_with_str_cohorts(
    fixture, api: AnophelesFstAnalysis, cohorts
):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    region = random.choice(api.contigs)
    site_mask = random.choice(api.site_mask_ids)
    fst_params = dict(
        region=region,
        cohorts=cohorts,
        sample_sets=all_sample_sets,
        site_mask=site_mask,
        min_cohort_size=1,
        n_jack=random.randint(10, 200),
    )

    # Run checks.
    check_pairwise_average_fst(api=api, fst_params=fst_params)


@parametrize_with_cases("fixture,api", cases=".")
def test_pairwise_average_fst_with_min_cohort_size(fixture, api: AnophelesFstAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    region = random.choice(api.contigs)
    site_mask = random.choice(api.site_mask_ids)
    cohorts = "admin1_year"
    fst_params = dict(
        region=region,
        cohorts=cohorts,
        sample_sets=all_sample_sets,
        site_mask=site_mask,
        min_cohort_size=15,
        n_jack=random.randint(10, 200),
    )

    # Run checks.
    check_pairwise_average_fst(api=api, fst_params=fst_params)


@parametrize_with_cases("fixture,api", cases=".")
def test_pairwise_average_fst_with_dict_cohorts(fixture, api: AnophelesFstAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_countries = api.sample_metadata()["country"].dropna().unique().tolist()
    cohorts = {country: f"country == '{country}'" for country in all_countries}
    region = random.choice(api.contigs)
    site_mask = random.choice(api.site_mask_ids)
    fst_params = dict(
        region=region,
        cohorts=cohorts,
        sample_sets=all_sample_sets,
        site_mask=site_mask,
        min_cohort_size=1,
        n_jack=random.randint(10, 200),
    )

    # Run checks.
    check_pairwise_average_fst(api=api, fst_params=fst_params)


@parametrize_with_cases("fixture,api", cases=".")
def test_pairwise_average_fst_with_sample_query(fixture, api: AnophelesFstAnalysis):
    # Set up test parameters.
    all_taxa = api.sample_metadata()["taxon"].dropna().unique().tolist()
    taxon = random.choice(all_taxa)
    sample_query = f"taxon == '{taxon}'"
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    cohorts = "admin2_month"
    region = random.choice(api.contigs)
    site_mask = random.choice(api.site_mask_ids)
    fst_params = dict(
        region=region,
        cohorts=cohorts,
        sample_sets=all_sample_sets,
        sample_query=sample_query,
        site_mask=site_mask,
        min_cohort_size=1,
        n_jack=random.randint(10, 200),
    )

    # Run checks.
    check_pairwise_average_fst(api=api, fst_params=fst_params)


@parametrize_with_cases("fixture,api", cases=".")
def test_pairwise_average_fst_with_bad_cohorts(fixture, api: AnophelesFstAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    cohorts = "foobar"
    region = random.choice(api.contigs)
    site_mask = random.choice(api.site_mask_ids)
    fst_params = dict(
        region=region,
        cohorts=cohorts,
        sample_sets=all_sample_sets,
        site_mask=site_mask,
        min_cohort_size=1,
    )

    # Run function under test.
    with pytest.raises(ValueError):
        api.pairwise_average_fst(**fst_params)
