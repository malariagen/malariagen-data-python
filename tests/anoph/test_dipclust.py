import random
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.dipclust import AnophelesDipClustAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesDipClustAnalysis(
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
    return AnophelesDipClustAnalysis(
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


@pytest.mark.parametrize("distance_metric", ["cityblock", "euclidean"])
@parametrize_with_cases("fixture,api", cases=".")
def test_plot_diplotype_clustering(
    fixture, api: AnophelesDipClustAnalysis, distance_metric
):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    linkage_methods = (
        "single",
        "complete",
        "average",
        "weighted",
        "centroid",
        "median",
        "ward",
    )
    sample_queries = (None, "sex_call == 'F'")
    dipclust_params = dict(
        region=fixture.random_region_str(region_size=5000),
        sample_sets=[random.choice(all_sample_sets)],
        linkage_method=random.choice(linkage_methods),
        distance_metric=distance_metric,
        sample_query=random.choice(sample_queries),
        show=False,
    )

    # Run checks.
    api.plot_diplotype_clustering(**dipclust_params)
