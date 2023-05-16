import pytest
import zarr
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.snp_data import AnophelesSnpData


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesSnpData(
        url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        gcs_url=_ag3.GCS_URL,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
        gff_gene_type="gene",
        gff_default_attributes=("ID", "Parent", "Name", "description"),
        default_site_mask="gamb_colu_arab",
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesSnpData(
        url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        gcs_url=_af1.GCS_URL,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        default_site_mask=("funestus",),
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
def test_open_snp_sites(fixture, api: AnophelesSnpData):
    root = api.open_snp_sites()
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in api.contigs:
        assert contig in root
        contig_grp = root[contig]
        assert "variants" in contig_grp
        variants_grp = contig_grp["variants"]
        assert "POS" in variants_grp
        assert "REF" in variants_grp
        assert "ALT" in variants_grp


def test_site_mask_ids_ag3(ag3_sim_api: AnophelesSnpData):
    assert ag3_sim_api.site_mask_ids == ("gamb_colu_arab", "gamb_colu", "arab")


def test_site_mask_ids_af1(af1_sim_api: AnophelesSnpData):
    assert af1_sim_api.site_mask_ids == ("funestus",)


@parametrize_with_cases("fixture,api", cases=".")
def test_open_site_filters(fixture, api: AnophelesSnpData):
    for mask in api.site_mask_ids:
        root = api.open_site_filters(mask=mask)
        assert isinstance(root, zarr.hierarchy.Group)
        for contig in api.contigs:
            assert contig in root
            contig_grp = root[contig]
            assert "variants" in contig_grp
            variants_grp = contig_grp["variants"]
            assert "filter_pass" in variants_grp
            filter_pass = variants_grp["filter_pass"]
            assert filter_pass.dtype == bool
