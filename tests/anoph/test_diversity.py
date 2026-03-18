import random

import pandas as pd
import plotly.graph_objects as go
import pytest
from malariagen_data import adir1 as _adir1
from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data import amin1 as _amin1
from malariagen_data.anoph.diversity import AnophelesDiversityAnalysis
from pytest_cases import parametrize_with_cases


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesDiversityAnalysis(
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
        default_site_mask="gamb_colu_arab",
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_ag3.TAXON_COLORS,
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesDiversityAnalysis(
        url=af1_sim_fixture.url,
        public_url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
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


@pytest.fixture
def adir1_sim_api(adir1_sim_fixture):
    return AnophelesDiversityAnalysis(
        url=adir1_sim_fixture.url,
        public_url=adir1_sim_fixture.url,
        config_path=_adir1.CONFIG_PATH,
        major_version_number=_adir1.MAJOR_VERSION_NUMBER,
        major_version_path=_adir1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        default_site_mask="dirus",
        results_cache=adir1_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_adir1.TAXON_COLORS,
    )


@pytest.fixture
def amin1_sim_api(amin1_sim_fixture):
    return AnophelesDiversityAnalysis(
        url=amin1_sim_fixture.url,
        public_url=amin1_sim_fixture.url,
        config_path=_amin1.CONFIG_PATH,
        major_version_number=_amin1.MAJOR_VERSION_NUMBER,
        major_version_path=_amin1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        default_site_mask="minimus",
        results_cache=amin1_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_amin1.TAXON_COLORS,
    )


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


def case_adir1_sim(adir1_sim_fixture, adir1_sim_api):
    return adir1_sim_fixture, adir1_sim_api


def case_amin1_sim(amin1_sim_fixture, amin1_sim_api):
    return amin1_sim_fixture, amin1_sim_api


@parametrize_with_cases("fixture,api", cases=".")
def test_cohort_diversity_stats(fixture, api: AnophelesDiversityAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = random.choice(all_sample_sets)
    df_samples = api.sample_metadata(sample_sets=[sample_set])
    cohort_sample_ids = df_samples["sample_id"].head(10).to_list()
    cohort_size = min(5, len(cohort_sample_ids))
    if cohort_size < 2:
        pytest.skip("not enough samples in simulated cohort")
    region = random.choice(api.contigs)
    site_mask = random.choice(api.site_mask_ids)

    diversity_params = dict(
        cohort=("test_cohort", f"sample_id in {cohort_sample_ids!r}"),
        cohort_size=cohort_size,
        region=region,
        sample_sets=[sample_set],
        site_mask=site_mask,
        n_jack=random.randint(10, 200),
    )

    # Run function under test.
    series = api.cohort_diversity_stats(**diversity_params)

    # Check results.
    assert isinstance(series, pd.Series)
    for field in [
        "theta_pi",
        "theta_pi_estimate",
        "theta_pi_ci_low",
        "theta_pi_ci_upp",
        "theta_w",
        "theta_w_estimate",
        "theta_w_ci_low",
        "theta_w_ci_upp",
        "tajima_d",
        "tajima_d_estimate",
        "tajima_d_ci_low",
        "tajima_d_ci_upp",
    ]:
        assert field in series.index
    assert series["theta_pi"] >= 0
    assert series["theta_w"] >= 0
    assert (
        series["tajima_d_ci_low"]
        <= series["tajima_d_estimate"]
        <= series["tajima_d_ci_upp"]
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_diversity_stats(fixture, api: AnophelesDiversityAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    cohorts = "admin1_year"
    region = random.choice(api.contigs)
    site_mask = random.choice(api.site_mask_ids)

    diversity_params = dict(
        cohorts=cohorts,
        cohort_size=5,
        region=region,
        sample_sets=all_sample_sets,
        site_mask=site_mask,
        n_jack=random.randint(10, 200),
    )

    # Run function under test.
    df_stats = api.diversity_stats(**diversity_params)

    # Check results.
    assert isinstance(df_stats, pd.DataFrame)
    assert len(df_stats) > 0
    for col in [
        "cohort",
        "theta_pi",
        "theta_pi_estimate",
        "theta_pi_ci_low",
        "theta_pi_ci_upp",
        "theta_w",
        "theta_w_estimate",
        "theta_w_ci_low",
        "theta_w_ci_upp",
        "tajima_d",
        "tajima_d_estimate",
        "tajima_d_ci_low",
        "tajima_d_ci_upp",
    ]:
        assert col in df_stats.columns
    assert (df_stats["theta_pi"] >= 0).all()
    assert (df_stats["theta_w"] >= 0).all()
    assert (df_stats["tajima_d_ci_low"] <= df_stats["tajima_d_estimate"]).all()
    assert (df_stats["tajima_d_estimate"] <= df_stats["tajima_d_ci_upp"]).all()


@parametrize_with_cases("fixture,api", cases=".")
def test_diversity_stats_with_bad_cohorts(fixture, api: AnophelesDiversityAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    region = random.choice(api.contigs)

    # Run function under test.
    with pytest.raises(ValueError):
        api.diversity_stats(
            cohorts="foobar",
            cohort_size=5,
            region=region,
            sample_sets=all_sample_sets,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_cohort_diversity_stats_with_bad_cohort_type(
    fixture, api: AnophelesDiversityAnalysis
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    with pytest.raises(TypeError):
        api.cohort_diversity_stats(
            cohort=12345,
            cohort_size=5,
            region=random.choice(api.contigs),
            sample_sets=all_sample_sets,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_diversity_stats(fixture, api: AnophelesDiversityAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    cohorts = "admin1_year"
    region = random.choice(api.contigs)
    site_mask = random.choice(api.site_mask_ids)

    df_stats = api.diversity_stats(
        cohorts=cohorts,
        cohort_size=5,
        region=region,
        sample_sets=all_sample_sets,
        site_mask=site_mask,
        n_jack=random.randint(10, 200),
    )

    if len(df_stats) == 0:
        pytest.skip("no cohorts with enough samples for cohort_size")
        return

    # Run function under test.
    figures = api.plot_diversity_stats(df_stats, show=False)

    # Check results.
    assert isinstance(figures, tuple)
    assert len(figures) == 4
    for fig in figures:
        assert isinstance(fig, go.Figure)
