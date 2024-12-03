import random
import pytest
from pytest_cases import parametrize_with_cases
import numpy as np
from numpy.testing import assert_allclose
import bokeh.models

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.h12 import AnophelesH12Analysis, haplotype_frequencies


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesH12Analysis(
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
        default_phasing_analysis="gamb_colu_arab",
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesH12Analysis(
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


def test_haplotype_frequencies():
    h1 = np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0],
        ],
        dtype="i1",
    )

    f = haplotype_frequencies(h1)
    assert isinstance(f, dict)
    vals = np.array(list(f.values()))
    vals.sort()
    assert np.all(vals >= 0)
    assert np.all(vals <= 1)
    assert_allclose(vals, np.array([0.2, 0.2, 0.2, 0.4]))


@parametrize_with_cases("fixture,api", cases=".")
def test_h12_calibration(fixture, api: AnophelesH12Analysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    window_sizes = np.random.randint(100, 500, size=random.randint(2, 5)).tolist()
    window_sizes = sorted(set([int(x) for x in window_sizes]))
    h12_params = dict(
        contig=random.choice(api.contigs),
        sample_sets=[random.choice(all_sample_sets)],
        window_sizes=window_sizes,
        min_cohort_size=5,
    )

    # Run function under test.
    calibration_runs = api.h12_calibration(**h12_params)

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
    fig = api.plot_h12_calibration(**h12_params, show=False)
    assert isinstance(fig, bokeh.models.Plot)


def check_h12_gwss(*, api, h12_params):
    # Run main gwss function under test.

    x, h12, contigs = api.h12_gwss(**h12_params)

    # Check results.
    assert isinstance(x, np.ndarray)
    assert isinstance(h12, np.ndarray)
    assert isinstance(contigs, np.ndarray)
    assert x.ndim == 1
    assert h12.ndim == 1
    assert contigs.ndim == 1
    assert x.shape == h12.shape
    assert x.shape == contigs.shape
    assert x.dtype.kind == "f"
    assert h12.dtype.kind == "f"
    assert contigs.dtype.kind == "i"
    assert np.all(h12 >= 0)
    assert np.all(h12 <= 1)

    # Check plotting functions.
    fig = api.plot_h12_gwss_track(**h12_params, show=False)
    assert isinstance(fig, bokeh.models.Plot)
    fig = api.plot_h12_gwss(**h12_params, contig_colors=["black", "red"], show=False)
    assert isinstance(fig, bokeh.models.GridPlot)


def check_h12_gwss_multi(*, api, h12_params):
    fig = api.plot_h12_gwss_multi_overlay(**h12_params, show=False)
    assert isinstance(fig, bokeh.models.GridPlot)
    fig = api.plot_h12_gwss_multi_panel(**h12_params, show=False)
    assert isinstance(fig, bokeh.models.GridPlot)


@parametrize_with_cases("fixture,api", cases=".")
def test_h12_gwss_with_default_analysis(fixture, api: AnophelesH12Analysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    h12_params = dict(
        contig=random.choice(api.contigs),
        sample_sets=[random.choice(all_sample_sets)],
        window_size=random.randint(100, 500),
        min_cohort_size=5,
    )

    # Run checks.
    check_h12_gwss(api=api, h12_params=h12_params)


@parametrize_with_cases("fixture,api", cases=".")
def test_h12_gwss_with_analysis(fixture, api: AnophelesH12Analysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = [random.choice(all_sample_sets)]
    contig = random.choice(api.contigs)
    window_size = random.randint(100, 500)

    for analysis in api.phasing_analysis_ids:
        # Check if any samples available for the given phasing analysis.
        try:
            ds_hap = api.haplotypes(
                sample_sets=sample_sets, analysis=analysis, region=contig
            )

        except ValueError:
            # No samples available, check similar error raised from H12.
            with pytest.raises(ValueError):
                api.h12_gwss(
                    contig=contig,
                    sample_sets=sample_sets,
                    analysis=analysis,
                    window_size=window_size,
                    min_cohort_size=1,
                )

        else:
            # Samples are available, run full checks.
            n_samples = ds_hap.sizes["samples"]
            h12_params = dict(
                contig=contig,
                sample_sets=sample_sets,
                analysis=analysis,
                window_size=window_size,
                min_cohort_size=n_samples,
            )
            check_h12_gwss(api=api, h12_params=h12_params)

            # Check min_cohort_size behaviour.
            with pytest.raises(ValueError):
                api.h12_gwss(
                    contig=contig,
                    sample_sets=sample_sets,
                    analysis=analysis,
                    window_size=window_size,
                    min_cohort_size=n_samples + 1,
                )


@parametrize_with_cases("fixture,api", cases=".")
def test_h12_gwss_multi_with_default_analysis(fixture, api: AnophelesH12Analysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_countries = api.sample_metadata()["country"].unique().tolist()
    country1, country2 = random.sample(all_countries, 2)
    cohort1_query = f"country == '{country1}'"
    cohort2_query = f"country == '{country2}'"
    h12_params = dict(
        contig=random.choice(api.contigs),
        sample_sets=all_sample_sets,
        window_size=random.randint(100, 500),
        min_cohort_size=1,
        cohorts={"cohort1": cohort1_query, "cohort2": cohort2_query},
    )

    # Run checks.
    check_h12_gwss_multi(api=api, h12_params=h12_params)


@parametrize_with_cases("fixture,api", cases=".")
def test_h12_gwss_multi_with_window_size_dict(fixture, api: AnophelesH12Analysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_countries = api.sample_metadata()["country"].unique().tolist()
    country1, country2 = random.sample(all_countries, 2)
    cohort1_query = f"country == '{country1}'"
    cohort2_query = f"country == '{country2}'"
    h12_params = dict(
        contig=random.choice(api.contigs),
        sample_sets=all_sample_sets,
        window_size={
            "cohort1": random.randint(100, 500),
            "cohort2": random.randint(100, 500),
        },
        min_cohort_size=1,
        cohorts={"cohort1": cohort1_query, "cohort2": cohort2_query},
    )

    # Run checks.
    check_h12_gwss_multi(api=api, h12_params=h12_params)


@parametrize_with_cases("fixture,api", cases=".")
def test_h12_gwss_multi_with_analysis(fixture, api: AnophelesH12Analysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_countries = api.sample_metadata()["country"].unique().tolist()
    country1, country2 = random.sample(all_countries, 2)
    cohort1_query = f"country == '{country1}'"
    cohort2_query = f"country == '{country2}'"
    contig = random.choice(api.contigs)

    for analysis in api.phasing_analysis_ids:
        # Check if any samples available for the given phasing analysis.
        try:
            ds_hap1 = api.haplotypes(
                sample_sets=all_sample_sets,
                sample_query=cohort1_query,
                analysis=analysis,
                region=contig,
            )
        except ValueError:
            n1 = 0
        else:
            n1 = ds_hap1.sizes["samples"]
        try:
            ds_hap2 = api.haplotypes(
                sample_sets=all_sample_sets,
                sample_query=cohort2_query,
                analysis=analysis,
                region=contig,
            )
        except ValueError:
            n2 = 0
        else:
            n2 = ds_hap2.sizes["samples"]

        if n1 > 0 and n2 > 0:
            # Samples are available, run full checks.
            h12_params = dict(
                analysis=analysis,
                contig=contig,
                sample_sets=all_sample_sets,
                window_size=random.randint(100, 500),
                min_cohort_size=min(n1, n2),
                cohorts={"cohort1": cohort1_query, "cohort2": cohort2_query},
            )
            check_h12_gwss_multi(api=api, h12_params=h12_params)

            # Check min_cohort_size behaviour.
            params = h12_params.copy()
            params["min_cohort_size"] = n1 + 1
            with pytest.raises(ValueError):
                api.plot_h12_gwss_multi_overlay(**params)
            params = h12_params.copy()
            params["min_cohort_size"] = n2 + 1
            with pytest.raises(ValueError):
                api.plot_h12_gwss_multi_panel(**params)
