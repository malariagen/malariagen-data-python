import pytest
import zarr
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.hap_data import AnophelesHapData


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesHapData(
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
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
        default_phasing_analysis="gamb_colu_arab",
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesHapData(
        url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        gcs_url=_af1.GCS_URL,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        results_cache=af1_sim_fixture.results_cache_path.as_posix(),
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


def _check_haplotype_sites(root, api: AnophelesHapData):
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in api.contigs:
        assert contig in root
        contig_grp = root[contig]
        assert "variants" in contig_grp
        variants = contig_grp["variants"]
        assert "POS" in variants
        assert "REF" in variants
        assert "ALT" in variants


@parametrize_with_cases("fixture,api", cases=".")
def test_open_haplotype_sites(fixture, api: AnophelesHapData):
    # Test default analysis.
    root = api.open_haplotype_sites()
    _check_haplotype_sites(root, api)

    # Test specific analyses.
    for analysis in api.phasing_analysis_ids:
        root = api.open_haplotype_sites(analysis=analysis)
        _check_haplotype_sites(root, api)
