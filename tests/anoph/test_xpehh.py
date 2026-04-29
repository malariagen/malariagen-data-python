import bokeh.models
import numpy as np
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.xpehh import AnophelesXpehhAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesXpehhAnalysis(
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
        default_phasing_analysis="gamb_colu_arab",
        taxon_colors=_ag3.TAXON_COLORS,
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesXpehhAnalysis(
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
        default_phasing_analysis="funestus",
        taxon_colors=_af1.TAXON_COLORS,
    )


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


def _setup_cohorts(api):
    """Helper to set up contig, sample_set, and cohort queries."""
    contig = str(np.random.choice(api.contigs))
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = str(np.random.choice(all_sample_sets))
    df_samples = api.sample_metadata(sample_sets=sample_set)

    if len(df_samples) < 2:
        pytest.skip("Not enough samples for two cohorts")

    sample_ids = df_samples["sample_id"].to_list()
    mid = len(sample_ids) // 2
    cohort1_query = f"sample_id in {sample_ids[:mid]}"
    cohort2_query = f"sample_id in {sample_ids[mid:]}"
    return contig, sample_set, cohort1_query, cohort2_query


def _safe_window_size(api, contig, sample_set, cohort1_query, cohort2_query):
    """Determine a safe window_size by first running without windowing."""
    x_raw, _ = api.xpehh_gwss(
        contig=contig,
        sample_sets=sample_set,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        window_size=None,
        min_cohort_size=2,
    )
    n_variants = len(x_raw)
    if n_variants < 2:
        pytest.skip(f"Only {n_variants} variants available, need at least 2")
    # Use half the available variants, clamped to [2, n_variants].
    return max(2, n_variants // 2)


@parametrize_with_cases("fixture,api", cases=".")
def test_xpehh_gwss(fixture, api: AnophelesXpehhAnalysis):
    contig, sample_set, cohort1_query, cohort2_query = _setup_cohorts(api)

    # Determine a safe window size from the actual data.
    window_size = _safe_window_size(
        api, contig, sample_set, cohort1_query, cohort2_query
    )

    # Run function under test with windowing.
    x, xpehh = api.xpehh_gwss(
        contig=contig,
        sample_sets=sample_set,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        window_size=window_size,
        min_cohort_size=2,
    )

    # Check results.
    assert isinstance(x, np.ndarray)
    assert isinstance(xpehh, np.ndarray)
    assert x.ndim == 1
    assert len(x) > 0
    assert len(x) == xpehh.shape[0]


@parametrize_with_cases("fixture,api", cases=".")
def test_xpehh_gwss_no_window(fixture, api: AnophelesXpehhAnalysis):
    contig, sample_set, cohort1_query, cohort2_query = _setup_cohorts(api)

    # Run function under test with no windowing.
    x, xpehh = api.xpehh_gwss(
        contig=contig,
        sample_sets=sample_set,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        window_size=None,
        min_cohort_size=2,
    )

    # Check results.
    assert isinstance(x, np.ndarray)
    assert isinstance(xpehh, np.ndarray)
    assert x.ndim == 1
    assert xpehh.ndim == 1
    assert len(x) == len(xpehh)


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_xpehh_gwss_track(fixture, api: AnophelesXpehhAnalysis):
    contig, sample_set, cohort1_query, cohort2_query = _setup_cohorts(api)

    window_size = _safe_window_size(
        api, contig, sample_set, cohort1_query, cohort2_query
    )

    # Run function under test.
    fig = api.plot_xpehh_gwss_track(
        contig=contig,
        sample_sets=sample_set,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        window_size=window_size,
        min_cohort_size=2,
        show=False,
    )

    # Check results.
    assert isinstance(fig, bokeh.models.Plot)


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_xpehh_gwss(fixture, api: AnophelesXpehhAnalysis):
    contig, sample_set, cohort1_query, cohort2_query = _setup_cohorts(api)

    window_size = _safe_window_size(
        api, contig, sample_set, cohort1_query, cohort2_query
    )

    # Run function under test.
    fig = api.plot_xpehh_gwss(
        contig=contig,
        sample_sets=sample_set,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        window_size=window_size,
        min_cohort_size=2,
        show=False,
    )

    # Check results.
    assert isinstance(fig, bokeh.models.GridPlot)
