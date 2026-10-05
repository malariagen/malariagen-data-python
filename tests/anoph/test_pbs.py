import pytest
from pytest_cases import parametrize_with_cases
import numpy as np
import bokeh.models

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data import adir1 as _adir1

from malariagen_data.anoph.pbs import AnophelesPbsAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesPbsAnalysis(
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
    return AnophelesPbsAnalysis(
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
    return AnophelesPbsAnalysis(
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


def case_adir1_sim(adir1_sim_fixture, adir1_sim_api):
    return adir1_sim_fixture, adir1_sim_api


@parametrize_with_cases("fixture,api", cases=".")
def test_pbs_gwss(fixture, api: AnophelesPbsAnalysis):
    # Set up test parameters - need 3 distinct cohorts.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_countries = api.sample_metadata()["country"].dropna().unique().tolist()
    if len(all_countries) < 3:
        pytest.skip("Not enough distinct countries for PBS test (need 3).")
    countries = np.random.choice(all_countries, size=3, replace=False).tolist()
    cohort1_query = f"country == {countries[0]!r}"
    cohort2_query = f"country == {countries[1]!r}"
    cohort3_query = f"country == {countries[2]!r}"
    pbs_params = dict(
        contig=str(np.random.choice(api.contigs)),
        sample_sets=all_sample_sets,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        cohort3_query=cohort3_query,
        site_mask=str(np.random.choice(api.site_mask_ids)),
        window_size=int(np.random.randint(10, 51)),
        min_cohort_size=1,
    )

    # Run main gwss function under test.
    x, pbs = api.pbs_gwss(**pbs_params)

    # Check results.
    assert isinstance(x, np.ndarray)
    assert isinstance(pbs, np.ndarray)
    assert x.ndim == 1
    assert pbs.ndim == 1
    assert x.shape == pbs.shape
    assert x.dtype.kind == "f"
    assert pbs.dtype.kind == "f"

    # Check plotting functions.
    fig = api.plot_pbs_gwss_track(**pbs_params, show=False)
    assert isinstance(fig, bokeh.models.Plot)
    fig = api.plot_pbs_gwss(**pbs_params, show=False)
    assert isinstance(fig, bokeh.models.GridPlot)


@parametrize_with_cases("fixture,api", cases=".")
def test_pbs_gwss_normed(fixture, api: AnophelesPbsAnalysis):
    # Test both normed=True and normed=False.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_countries = api.sample_metadata()["country"].dropna().unique().tolist()
    if len(all_countries) < 3:
        pytest.skip("Not enough distinct countries for PBS test (need 3).")
    countries = np.random.choice(all_countries, size=3, replace=False).tolist()
    cohort1_query = f"country == {countries[0]!r}"
    cohort2_query = f"country == {countries[1]!r}"
    cohort3_query = f"country == {countries[2]!r}"
    common_params = dict(
        contig=str(np.random.choice(api.contigs)),
        sample_sets=all_sample_sets,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        cohort3_query=cohort3_query,
        site_mask=str(np.random.choice(api.site_mask_ids)),
        window_size=int(np.random.randint(10, 51)),
        min_cohort_size=1,
    )

    # Run with normed=True.
    x_normed, pbs_normed = api.pbs_gwss(**common_params, normed=True)
    assert isinstance(x_normed, np.ndarray)
    assert isinstance(pbs_normed, np.ndarray)
    assert x_normed.shape == pbs_normed.shape

    # Run with normed=False.
    x_unnormed, pbs_unnormed = api.pbs_gwss(**common_params, normed=False)
    assert isinstance(x_unnormed, np.ndarray)
    assert isinstance(pbs_unnormed, np.ndarray)
    assert x_unnormed.shape == pbs_unnormed.shape


@parametrize_with_cases("fixture,api", cases=".")
def test_pbs_gwss_window_size_too_large(fixture, api: AnophelesPbsAnalysis):
    # When window_size exceeds available SNPs, a UserWarning must be issued and
    # the function must still return a valid result using the adjusted window_size.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_countries = api.sample_metadata()["country"].dropna().unique().tolist()
    if len(all_countries) < 3:
        pytest.skip("Not enough distinct countries for PBS test (need 3).")
    countries = np.random.choice(all_countries, size=3, replace=False).tolist()
    cohort1_query = f"country == {countries[0]!r}"
    cohort2_query = f"country == {countries[1]!r}"
    cohort3_query = f"country == {countries[2]!r}"
    with pytest.warns(UserWarning, match="window_size"):
        x, pbs = api.pbs_gwss(
            contig=str(np.random.choice(api.contigs)),
            sample_sets=all_sample_sets,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            cohort3_query=cohort3_query,
            site_mask=str(np.random.choice(api.site_mask_ids)),
            window_size=10_000_000,  # far larger than any fixture SNP count
            min_cohort_size=1,
        )
    assert isinstance(x, np.ndarray)
    assert isinstance(pbs, np.ndarray)
    assert len(x) > 0
    assert x.shape == pbs.shape


@parametrize_with_cases("fixture,api", cases=".")
def test_pbs_gwss_too_few_snps(fixture, api: AnophelesPbsAnalysis):
    # When min_snps_threshold exceeds available SNPs, a ValueError must be raised.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_countries = api.sample_metadata()["country"].dropna().unique().tolist()
    if len(all_countries) < 3:
        pytest.skip("Not enough distinct countries for PBS test (need 3).")
    countries = np.random.choice(all_countries, size=3, replace=False).tolist()
    cohort1_query = f"country == {countries[0]!r}"
    cohort2_query = f"country == {countries[1]!r}"
    cohort3_query = f"country == {countries[2]!r}"
    with pytest.raises(ValueError, match="Too few SNP sites"):
        api.pbs_gwss(
            contig=str(np.random.choice(api.contigs)),
            sample_sets=all_sample_sets,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            cohort3_query=cohort3_query,
            site_mask=str(np.random.choice(api.site_mask_ids)),
            window_size=100,
            min_cohort_size=1,
            min_snps_threshold=10_000_000,
        )
