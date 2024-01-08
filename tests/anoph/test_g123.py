import random
import pytest
from pytest_cases import parametrize_with_cases
import numpy as np
import bokeh.models

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.g123 import AnophelesG123Analysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesG123Analysis(
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
        gff_default_attributes=("ID", "Parent", "Name", "description"),
        default_site_mask="gamb_colu_arab",
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_ag3.TAXON_COLORS,
        default_phasing_analysis="gamb_colu_arab",
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesG123Analysis(
        url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        gcs_url=_af1.GCS_URL,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        default_site_mask="funestus",
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


def check_g123_gwss_results(*, x, g123):
    assert isinstance(x, np.ndarray)
    assert isinstance(g123, np.ndarray)
    assert x.ndim == 1
    assert g123.ndim == 1
    assert x.shape == g123.shape
    assert x.dtype.kind == "f"
    assert g123.dtype.kind == "f"
    assert np.all(g123 >= 0)
    assert np.all(g123 <= 1)


@parametrize_with_cases("fixture,api", cases=".")
def test_g123_gwss_with_phased_sites(fixture, api: AnophelesG123Analysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    g123_params = dict(
        contig=random.choice(api.contigs),
        sites=random.choice(api.phasing_analysis_ids),
        sample_sets=[random.choice(all_sample_sets)],
        window_size=random.randint(100, 500),
        min_cohort_size=10,
    )

    # Run function under test.
    x, g123 = api.g123_gwss(**g123_params)

    # Check results.
    check_g123_gwss_results(x=x, g123=g123)

    # Run plotting functions.
    fig = api.plot_g123_gwss_track(**g123_params, show=False)
    assert isinstance(fig, bokeh.models.Plot)
    fig = api.plot_g123_gwss(**g123_params, show=False)
    assert isinstance(fig, bokeh.models.GridPlot)


@parametrize_with_cases("fixture,api", cases=".")
def test_g123_gwss_with_segregating_sites(fixture, api: AnophelesG123Analysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    g123_params = dict(
        contig=random.choice(api.contigs),
        sites="segregating",
        site_mask=random.choice(api.site_mask_ids),
        sample_sets=[random.choice(all_sample_sets)],
        window_size=random.randint(100, 500),
        min_cohort_size=10,
    )

    # Run function under test.
    x, g123 = api.g123_gwss(**g123_params)

    # Check results.
    check_g123_gwss_results(x=x, g123=g123)

    # Run plotting functions.
    fig = api.plot_g123_gwss_track(**g123_params, show=False)
    assert isinstance(fig, bokeh.models.Plot)
    fig = api.plot_g123_gwss(**g123_params, show=False)
    assert isinstance(fig, bokeh.models.GridPlot)


@parametrize_with_cases("fixture,api", cases=".")
def test_g123_gwss_with_all_sites(fixture, api: AnophelesG123Analysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    g123_params = dict(
        contig=random.choice(api.contigs),
        sites="all",
        site_mask=None,
        sample_sets=[random.choice(all_sample_sets)],
        window_size=random.randint(100, 500),
        min_cohort_size=10,
    )

    # Run function under test.
    x, g123 = api.g123_gwss(**g123_params)

    # Check results.
    check_g123_gwss_results(x=x, g123=g123)

    # Run plotting functions.
    fig = api.plot_g123_gwss_track(**g123_params, show=False)
    assert isinstance(fig, bokeh.models.Plot)
    fig = api.plot_g123_gwss(**g123_params, show=False)
    assert isinstance(fig, bokeh.models.GridPlot)


@parametrize_with_cases("fixture,api", cases=".")
def test_g123_calibration(fixture, api: AnophelesG123Analysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    window_sizes = np.random.randint(100, 500, size=random.randint(2, 5)).tolist()
    window_sizes = sorted([int(x) for x in window_sizes])
    g123_params = dict(
        contig=random.choice(api.contigs),
        sites=random.choice(api.phasing_analysis_ids),
        sample_sets=[random.choice(all_sample_sets)],
        min_cohort_size=10,
        window_sizes=window_sizes,
    )

    # Run function under test.
    calibration_runs = api.g123_calibration(**g123_params)

    # Check results.
    assert isinstance(calibration_runs, dict)
    assert len(calibration_runs) == len(window_sizes)
    assert list(calibration_runs.keys()) == [str(win) for win in window_sizes]
    for w in window_sizes:
        x = calibration_runs[str(w)]
        assert isinstance(x, np.ndarray)
        assert np.all(x >= 0)
        assert np.all(x <= 1)

    # Run plotting function.
    fig = api.plot_g123_calibration(**g123_params, show=False)
    assert isinstance(fig, bokeh.models.Plot)
