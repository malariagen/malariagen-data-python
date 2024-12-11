import random

import pytest
from pytest_cases import parametrize_with_cases
import plotly.graph_objects as go  # type: ignore

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.snp_frq import AnophelesSnpFrequencyAnalysis

from .test_snp_frq import random_transcript


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesSnpFrequencyAnalysis(
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
        default_site_mask="gamb_colu_arab",
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_ag3.TAXON_COLORS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesSnpFrequencyAnalysis(
        url=af1_sim_fixture.url,
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
def test_plot_frequencies_heatmap(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice(api.site_mask_ids + (None,))
    min_cohort_size = random.randint(0, 2)
    transcript = random_transcript(api=api).name
    cohorts = random.choice(
        ["admin1_year", "admin1_month", "admin2_year", "admin2_month"]
    )

    # Set up call params.
    params = dict(
        transcript=transcript,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
        sample_sets=sample_sets,
    )

    # Test SNP allele frequencies.
    df_snp = api.snp_allele_frequencies(**params)
    fig = api.plot_frequencies_heatmap(df_snp, show=False, max_len=None)
    assert isinstance(fig, go.Figure)

    # Test amino acid change allele frequencies.
    df_aa = api.aa_allele_frequencies(**params)
    fig = api.plot_frequencies_heatmap(df_aa, show=False, max_len=None)
    assert isinstance(fig, go.Figure)

    # Test max_len behaviour.
    with pytest.raises(ValueError):
        api.plot_frequencies_heatmap(df_snp, show=False, max_len=len(df_snp) - 1)

    # Test index parameter - if None, should use dataframe index.
    fig = api.plot_frequencies_heatmap(df_snp, show=False, index=None, max_len=None)
    # Not unique.
    with pytest.raises(ValueError):
        api.plot_frequencies_heatmap(df_snp, show=False, index="contig", max_len=None)


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_frequencies_time_series(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice(api.site_mask_ids + (None,))
    min_cohort_size = random.randint(0, 2)
    transcript = random_transcript(api=api).name
    area_by = random.choice(["country", "admin1_iso", "admin2_name"])
    period_by = random.choice(["year", "quarter", "month"])

    # Compute SNP frequencies.
    ds = api.snp_allele_frequencies_advanced(
        transcript=transcript,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
    )

    # Trim things down a bit for speed.
    ds = ds.isel(variants=slice(0, 100))

    # Plot.
    fig = api.plot_frequencies_time_series(ds, show=False)

    # Test.
    assert isinstance(fig, go.Figure)

    # Compute amino acid change frequencies.
    ds = api.aa_allele_frequencies_advanced(
        transcript=transcript,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        min_cohort_size=min_cohort_size,
    )

    # Trim things down a bit for speed.
    ds = ds.isel(variants=slice(0, 100))

    # Plot.
    fig = api.plot_frequencies_time_series(ds, show=False)

    # Test.
    assert isinstance(fig, go.Figure)


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_frequencies_time_series_with_taxa(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice(api.site_mask_ids + (None,))
    transcript = random_transcript(api=api).name
    area_by = random.choice(["country", "admin1_iso", "admin2_name"])
    period_by = random.choice(["year", "quarter", "month"])

    # Pick a random taxon and taxa from valid taxa.
    sample_sets_taxa = (
        api.sample_metadata(sample_sets=sample_sets)["taxon"].dropna().unique().tolist()
    )
    taxon = random.choice(sample_sets_taxa)
    taxa = random.sample(sample_sets_taxa, random.randint(1, len(sample_sets_taxa)))

    # Compute SNP frequencies.
    ds = api.snp_allele_frequencies_advanced(
        transcript=transcript,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        min_cohort_size=1,  # Don't exclude any samples.
        site_mask=site_mask,
    )

    # Trim things down a bit for speed.
    ds = ds.isel(variants=slice(0, 100))

    # Plot with taxon.
    fig = api.plot_frequencies_time_series(ds, show=False, taxa=taxon)

    # Test taxon plot.
    assert isinstance(fig, go.Figure)

    # Plot with taxa.
    fig = api.plot_frequencies_time_series(ds, show=False, taxa=taxa)

    # Test taxa plot.
    assert isinstance(fig, go.Figure)


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_frequencies_time_series_with_areas(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice(api.site_mask_ids + (None,))
    transcript = random_transcript(api=api).name
    area_by = random.choice(["country", "admin1_iso", "admin2_name"])
    period_by = random.choice(["year", "quarter", "month"])

    # Compute SNP frequencies.
    ds = api.snp_allele_frequencies_advanced(
        transcript=transcript,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        min_cohort_size=1,  # Don't exclude any samples.
        site_mask=site_mask,
    )

    # Trim things down a bit for speed.
    ds = ds.isel(variants=slice(0, 100))

    # Extract cohorts into a DataFrame.
    cohort_vars = [v for v in ds if str(v).startswith("cohort_")]
    df_cohorts = ds[cohort_vars].to_dataframe()

    # Pick a random area and areas from valid areas.
    cohorts_areas = df_cohorts["cohort_area"].dropna().unique().tolist()
    area = random.choice(cohorts_areas)
    areas = random.sample(cohorts_areas, random.randint(1, len(cohorts_areas)))

    # Plot with area.
    fig = api.plot_frequencies_time_series(ds, show=False, areas=area)

    # Test areas plot.
    assert isinstance(fig, go.Figure)

    # Plot with areas.
    fig = api.plot_frequencies_time_series(ds, show=False, areas=areas)

    # Test area plot.
    assert isinstance(fig, go.Figure)


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_frequencies_interactive_map(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    import ipywidgets  # type: ignore

    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice(api.site_mask_ids + (None,))
    min_cohort_size = random.randint(0, 2)
    transcript = random_transcript(api=api).name
    area_by = random.choice(["country", "admin1_iso", "admin2_name"])
    period_by = random.choice(["year", "quarter", "month"])

    # Compute SNP frequencies.
    ds = api.snp_allele_frequencies_advanced(
        transcript=transcript,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
    )

    # Trim things down a bit for speed.
    ds = ds.isel(variants=slice(0, 100))

    # Plot.
    fig = api.plot_frequencies_interactive_map(ds)

    # Test.
    assert isinstance(fig, ipywidgets.Widget)

    # Compute amino acid change frequencies.
    ds = api.aa_allele_frequencies_advanced(
        transcript=transcript,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        min_cohort_size=min_cohort_size,
    )

    # Trim things down a bit for speed.
    ds = ds.isel(variants=slice(0, 100))

    # Plot.
    fig = api.plot_frequencies_interactive_map(ds)

    # Test.
    assert isinstance(fig, ipywidgets.Widget)
