import random

import numpy as np
import plotly.graph_objects as go  # type: ignore
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.distance import AnophelesDistanceAnalysis
from malariagen_data.anoph import pca_params


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesDistanceAnalysis(
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
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesDistanceAnalysis(
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


def check_biallelic_diplotype_pairwise_distance(*, api, data_params, metric):
    # Check available data.
    ds = api.biallelic_snp_calls(**data_params)
    n_samples = ds.sizes["samples"]
    n_snps_available = ds.sizes["variants"]
    n_snps = random.randint(4, n_snps_available)

    # Run the distance computation.
    dist, samples, n_snps_used = api.biallelic_diplotype_pairwise_distances(
        n_snps=n_snps,
        metric=metric,
        **data_params,
    )

    # Check types.
    assert isinstance(dist, np.ndarray)
    assert isinstance(samples, np.ndarray)
    assert isinstance(n_snps_used, int)

    # Check sizes.
    assert dist.ndim == 1  # condensed form distance matrix
    assert dist.shape[0] == int((n_samples * (n_samples - 1)) / 2)
    assert samples.ndim == 1
    assert samples.shape[0] == n_samples
    assert n_snps_used >= n_snps
    assert n_snps_used <= n_snps_available

    # Check types.
    assert isinstance(dist, np.ndarray)
    assert isinstance(samples, np.ndarray)
    assert isinstance(n_snps_used, int)

    # Check sizes.
    assert dist.ndim == 1  # condensed form distance matrix
    assert dist.shape[0] == int((n_samples * (n_samples - 1)) / 2)
    assert samples.ndim == 1
    assert samples.shape[0] == n_samples
    assert n_snps_used >= n_snps
    assert n_snps_used <= n_snps_available


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_diplotype_pairwise_distance_with_metric(
    fixture, api: AnophelesDistanceAnalysis
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    data_params = dict(
        region=random.choice(api.contigs),
        sample_sets=random.sample(all_sample_sets, 2),
        site_mask=random.choice((None,) + api.site_mask_ids),
        min_minor_ac=pca_params.min_minor_ac_default,
        max_missing_an=pca_params.max_missing_an_default,
    )

    for metric in "cityblock", "euclidean", "sqeuclidean":
        check_biallelic_diplotype_pairwise_distance(
            api=api,
            data_params=data_params,
            metric=metric,
        )


def check_njt(*, api, data_params, metric, algorithm):
    # Check available data.
    ds = api.biallelic_snp_calls(**data_params)
    n_samples = ds.sizes["samples"]
    n_snps_available = ds.sizes["variants"]
    n_snps = random.randint(4, n_snps_available)

    # Run the distance computation.
    Z, samples, n_snps_used = api.njt(
        n_snps=n_snps,
        metric=metric,
        algorithm=algorithm,
        **data_params,
    )

    # Check types.
    assert isinstance(Z, np.ndarray)
    assert isinstance(samples, np.ndarray)
    assert isinstance(n_snps_used, int)

    # Check sizes.
    assert Z.ndim == 2  # njt linkage matrix
    assert Z.shape[0] == n_samples - 1  # number of internal nodes in the tree
    assert Z.shape[1] == 5
    assert samples.ndim == 1
    assert samples.shape[0] == n_samples
    assert n_snps_used >= n_snps
    assert n_snps_used <= n_snps_available


@parametrize_with_cases("fixture,api", cases=".")
def test_njt_with_metric(fixture, api: AnophelesDistanceAnalysis):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    data_params = dict(
        region=random.choice(api.contigs),
        sample_sets=random.sample(all_sample_sets, 2),
        site_mask=random.choice((None,) + api.site_mask_ids),
        min_minor_ac=pca_params.min_minor_ac_default,
        max_missing_an=pca_params.max_missing_an_default,
    )
    parametrize_metric = "cityblock", "euclidean", "sqeuclidean"
    algorithm = random.choice(["dynamic", "rapid", "canonical"])
    for metric in parametrize_metric:
        check_njt(
            api=api,
            data_params=data_params,
            metric=metric,
            algorithm=algorithm,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_njt_with_algorithm(fixture, api: AnophelesDistanceAnalysis):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    data_params = dict(
        region=random.choice(api.contigs),
        sample_sets=random.sample(all_sample_sets, 2),
        site_mask=random.choice((None,) + api.site_mask_ids),
        min_minor_ac=pca_params.min_minor_ac_default,
        max_missing_an=pca_params.max_missing_an_default,
    )
    metric = random.choice(["cityblock", "euclidean", "sqeuclidean"])
    parametrize_algorithm = "dynamic", "rapid", "canonical"
    for algorithm in parametrize_algorithm:
        check_njt(
            api=api,
            data_params=data_params,
            metric=metric,
            algorithm=algorithm,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_njt(fixture, api: AnophelesDistanceAnalysis):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    data_params = dict(
        region=random.choice(api.contigs),
        sample_sets=random.sample(all_sample_sets, 2),
        site_mask=random.choice((None,) + api.site_mask_ids),
        min_minor_ac=pca_params.min_minor_ac_default,
        max_missing_an=pca_params.max_missing_an_default,
    )
    metric = random.choice(["cityblock", "euclidean", "sqeuclidean"])
    algorithm = random.choice(["dynamic", "rapid", "canonical"])
    custom_cohorts = {
        "male": "sex_call == 'M'",
        "female": "sex_call == 'F'",
    }
    colors = [None, "taxon", "country", "admin1_year", custom_cohorts]
    symbols = ["country", None, custom_cohorts, "admin2_month", "taxon"]

    # Check available data.
    ds = api.biallelic_snp_calls(**data_params)
    n_snps_available = ds.sizes["variants"]
    n_snps = random.randint(4, n_snps_available)

    # Exercise the function.
    for color, symbol in zip(colors, symbols):
        fig = api.plot_njt(
            metric=metric,
            algorithm=algorithm,
            color=color,
            symbol=symbol,
            n_snps=n_snps,
            show=False,
            **data_params,
        )
        assert isinstance(fig, go.Figure)
