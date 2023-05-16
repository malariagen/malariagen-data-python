import random
from itertools import product

import dask.array as da
import numpy as np
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
        default_site_mask="funestus",
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
        variants = contig_grp["variants"]
        assert "POS" in variants
        assert "REF" in variants
        assert "ALT" in variants


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


@parametrize_with_cases("fixture,api", cases=".")
def test_open_snp_genotypes(fixture, api: AnophelesSnpData):
    for rec in api.sample_sets().itertuples():
        sample_set = rec.sample_set
        n_samples = rec.sample_count
        root = api.open_snp_genotypes(sample_set=sample_set)
        assert isinstance(root, zarr.hierarchy.Group)

        # Check samples array.
        assert "samples" in root
        samples = root["samples"][:]
        assert samples.ndim == 1
        assert samples.shape[0] == n_samples
        assert samples.dtype.kind == "S"

        for contig in api.contigs:
            assert contig in root
            contig_grp = root[contig]

            # Check calldata arrays.
            n_sites = fixture.n_sites[contig]
            assert "calldata" in contig_grp
            calldata = contig_grp["calldata"]
            assert "GT" in calldata
            gt = calldata["GT"]
            assert gt.shape == (n_sites, n_samples, 2)
            assert gt.dtype == "i1"
            assert "GQ" in calldata
            gq = calldata["GQ"]
            assert gq.shape == (n_sites, n_samples)
            assert gq.dtype == "i1"
            assert "MQ" in calldata
            mq = calldata["MQ"]
            assert mq.shape == (n_sites, n_samples)
            assert mq.dtype == "f4"
            assert "AD" in calldata
            ad = calldata["AD"]
            assert ad.shape == (n_sites, n_samples, 4)
            assert ad.dtype == "i2"


def _check_site_filters(api: AnophelesSnpData, mask, region):
    filter_pass = api.site_filters(region=region, mask=mask)
    assert isinstance(filter_pass, da.Array)
    assert filter_pass.ndim == 1
    assert filter_pass.dtype == bool


@parametrize_with_cases("fixture,api", cases=".")
def test_site_filters(fixture, api: AnophelesSnpData):
    for mask in api.site_mask_ids:
        # Test with contig.
        contig = random.choice(api.contigs)
        _check_site_filters(api, mask=mask, region=contig)

        # Test with region string.
        region = f"{contig}:20,000-50,000"
        _check_site_filters(api, mask=mask, region=region)

        # Test with genome feature ID.
        df_gff = api.genome_features(attributes=["ID"])
        region = random.choice(df_gff["ID"].dropna().to_list())
        _check_site_filters(api, mask=mask, region=region)


def _check_snp_sites(api: AnophelesSnpData, region):
    pos = api.snp_sites(region=region, field="POS")
    ref = api.snp_sites(region=region, field="REF")
    alt = api.snp_sites(region=region, field="ALT")
    assert isinstance(pos, da.Array)
    assert pos.ndim == 1
    assert pos.dtype == "i4"
    assert isinstance(ref, da.Array)
    assert ref.ndim == 1
    assert ref.dtype == "S1"
    assert isinstance(alt, da.Array)
    assert alt.ndim == 2
    assert alt.dtype == "S1"
    assert pos.shape[0] == ref.shape[0] == alt.shape[0]

    # Apply site mask.
    mask = random.choice(api.site_mask_ids)
    filter_pass = api.site_filters(region=region, mask=mask).compute()
    n_pass = np.count_nonzero(filter_pass)
    pos_pass = api.snp_sites(
        region=region,
        field="POS",
        site_mask=mask,
    )
    assert isinstance(pos_pass, da.Array)
    assert pos_pass.ndim == 1
    assert pos_pass.dtype == "i4"
    assert pos_pass.shape[0] == n_pass
    assert pos_pass.compute().shape == pos_pass.shape
    for f in "POS", "REF", "ALT":
        d = api.snp_sites(
            region=region,
            site_mask=mask,
            field=f,
        )
        assert isinstance(d, da.Array)
        assert d.shape[0] == n_pass
        assert d.shape == d.compute().shape


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_sites(fixture, api: AnophelesSnpData):
    # Test with contig.
    contig = random.choice(api.contigs)
    _check_snp_sites(api=api, region=contig)

    # Test with region string.
    region = f"{contig}:20,000-50,000"
    _check_snp_sites(api=api, region=region)

    # Test with genome feature ID.
    df_gff = api.genome_features(attributes=["ID"])
    region = random.choice(df_gff["ID"].dropna().to_list())
    _check_snp_sites(api=api, region=region)


def _check_snp_genotypes(api, sample_sets, region):
    df_samples = api.sample_metadata(sample_sets=sample_sets)

    # Check default field (GT).
    gt = api.snp_genotypes(region=region, sample_sets=sample_sets)
    assert isinstance(gt, da.Array)
    assert gt.ndim == 3
    assert gt.dtype == "i1"
    assert gt.shape[1] == len(df_samples)

    # Check GT.
    x = api.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        field="GT",
    )
    assert isinstance(x, da.Array)
    assert x.ndim == 3
    assert x.dtype == "i1"

    # Check GQ.
    x = api.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        field="GQ",
    )
    assert isinstance(x, da.Array)
    assert x.ndim == 2
    assert x.dtype == "i1"

    # Check MQ.
    x = api.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        field="MQ",
    )
    assert isinstance(x, da.Array)
    assert x.ndim == 2
    assert x.dtype == "f4"

    # Check AD.
    x = api.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        field="AD",
    )
    assert isinstance(x, da.Array)
    assert x.ndim == 3
    assert x.dtype == "i2"

    # Check with site mask.
    mask = random.choice(api.site_mask_ids)
    filter_pass = api.site_filters(region=region, mask=mask).compute()
    gt_pass = api.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        site_mask=mask,
    )
    assert isinstance(gt_pass, da.Array)
    assert gt_pass.ndim == 3
    assert gt_pass.dtype == "i1"
    assert gt_pass.shape[0] == np.count_nonzero(filter_pass)
    assert gt_pass.shape[1] == len(df_samples)
    assert gt_pass.shape[2] == 2

    # Check native versus auto chunks.
    gt_native = api.snp_genotypes(
        region=region, sample_sets=sample_sets, chunks="native"
    )
    gt_auto = api.snp_genotypes(region=region, sample_sets=sample_sets, chunks="auto")
    assert gt_native.chunks != gt_auto.chunks


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_genotypes(fixture, api: AnophelesSnpData):
    # Here we manually parametrize, because different parameters
    # need to be chosen at runtime.

    # Parametrize sample_sets.
    all_releases = api.releases
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    parametrize_sample_sets = [
        None,
        random.choice(all_sample_sets),
        np.random.choice(all_sample_sets, size=2, replace=False).tolist(),
        random.choice(all_releases),
    ]

    # Parametrize region.
    contig = random.choice(api.contigs)
    df_gff = api.genome_features(attributes=["ID"])
    parametrize_region = [
        contig,
        f"{contig}:20,000-50,000",
        random.choice(df_gff["ID"].dropna().to_list()),
    ]

    # Run tests.
    for sample_sets, region in product(parametrize_sample_sets, parametrize_region):
        _check_snp_genotypes(api=api, sample_sets=sample_sets, region=region)
