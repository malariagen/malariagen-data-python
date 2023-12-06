import random
from itertools import product

import bokeh.model
import dask.array as da
import numpy as np
import pytest
import xarray as xr
import zarr
from numpy.testing import assert_array_equal
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
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
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
        results_cache=af1_sim_fixture.results_cache_path.as_posix(),
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

        # Check calldata arrays.
        for contig in api.contigs:
            assert contig in root
            contig_grp = root[contig]

            n_sites = fixture.n_snp_sites[contig]
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


def check_site_filters(api: AnophelesSnpData, mask, region):
    filter_pass = api.site_filters(region=region, mask=mask)
    assert isinstance(filter_pass, da.Array)
    assert filter_pass.ndim == 1
    assert filter_pass.dtype == bool


@parametrize_with_cases("fixture,api", cases=".")
def test_site_filters(fixture, api: AnophelesSnpData):
    for mask in api.site_mask_ids:
        # Test with contig.
        contig = fixture.random_contig()
        check_site_filters(api, mask=mask, region=contig)

        # Test with region string.
        region = fixture.random_region_str()
        check_site_filters(api, mask=mask, region=region)

        # Test with genome feature ID.
        df_gff = api.genome_features(attributes=["ID"])
        region = random.choice(df_gff["ID"].dropna().to_list())
        check_site_filters(api, mask=mask, region=region)


def check_snp_sites(api: AnophelesSnpData, region):
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
    contig = fixture.random_contig()
    check_snp_sites(api=api, region=contig)

    # Test with region string.
    region = fixture.random_region_str()
    check_snp_sites(api=api, region=region)

    # Test with genome feature ID.
    df_gff = api.genome_features(attributes=["ID"])
    region = random.choice(df_gff["ID"].dropna().to_list())
    check_snp_sites(api=api, region=region)


@parametrize_with_cases("fixture,api", cases=".")
def test_open_site_annotations(fixture, api):
    root = api.open_site_annotations()
    assert isinstance(root, zarr.hierarchy.Group)
    for f in (
        "codon_degeneracy",
        "codon_nonsyn",
        "codon_position",
        "seq_cls",
        "seq_flen",
        "seq_relpos_start",
        "seq_relpos_stop",
    ):
        assert f in root
        for contig in api.contigs:
            assert contig in root[f]
            z = root[f][contig]
            # Zarr data should be aligned with genome sequence.
            assert z.shape == (len(api.genome_sequence(region=contig)),)


def _check_site_annotations(api: AnophelesSnpData, region, site_mask):
    ds_snp = api.snp_variants(region=region, site_mask=site_mask)
    n_variants = ds_snp.dims["variants"]
    ds_ann = api.site_annotations(region=region, site_mask=site_mask)
    # Site annotations dataset should be aligned with SNP sites.
    assert ds_ann.dims["variants"] == n_variants
    assert isinstance(ds_ann, xr.Dataset)
    for f in (
        "codon_degeneracy",
        "codon_nonsyn",
        "codon_position",
        "seq_cls",
        "seq_flen",
        "seq_relpos_start",
        "seq_relpos_stop",
    ):
        d = ds_ann[f]
        assert d.ndim == 1
        assert d.dims == ("variants",)
        assert d.shape == (n_variants,)


@parametrize_with_cases("fixture,api", cases=".")
def test_site_annotations(fixture, api):
    # Parametrize region.
    contig = fixture.random_contig()
    df_gff = api.genome_features(attributes=["ID"])
    # Don't need to support multiple regions at this time.
    parametrize_region = [
        contig,
        fixture.random_region_str(),
        random.choice(df_gff["ID"].dropna().to_list()),
    ]

    # Parametrize site_mask.
    parametrize_site_mask = (None, random.choice(api.site_mask_ids))

    # Run tests.
    for region, site_mask in product(
        parametrize_region,
        parametrize_site_mask,
    ):
        _check_site_annotations(
            api=api,
            region=region,
            site_mask=site_mask,
        )


def check_snp_genotypes(api, sample_sets, region):
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
def test_snp_genotypes_with_sample_sets_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    region = fixture.random_region_str()

    # Parametrize sample_sets.
    all_releases = api.releases
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    parametrize_sample_sets = [
        None,
        random.choice(all_sample_sets),
        random.sample(all_sample_sets, 2),
        random.choice(all_releases),
    ]

    # Run tests.
    for sample_sets in parametrize_sample_sets:
        check_snp_genotypes(api=api, sample_sets=sample_sets, region=region)


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_genotypes_with_region_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)

    # Parametrize region.
    contig = fixture.random_contig()
    df_gff = api.genome_features(attributes=["ID"])
    parametrize_region = [
        contig,
        fixture.random_region_str(),
        [fixture.random_region_str(), fixture.random_region_str()],
        random.choice(df_gff["ID"].dropna().to_list()),
    ]

    # Run tests.
    for region in parametrize_region:
        check_snp_genotypes(api=api, sample_sets=sample_sets, region=region)


def check_snp_calls(api, sample_sets, region, site_mask):
    ds = api.snp_calls(region=region, sample_sets=sample_sets, site_mask=site_mask)
    assert isinstance(ds, xr.Dataset)

    # Check fields.
    expected_data_vars = {
        "variant_allele",
        "call_genotype",
        "call_genotype_mask",
        "call_GQ",
        "call_AD",
        "call_MQ",
    }
    for m in api.site_mask_ids:
        expected_data_vars.add(f"variant_filter_pass_{m}")
    assert set(ds.data_vars) == expected_data_vars

    expected_coords = {
        "variant_contig",
        "variant_position",
        "sample_id",
    }
    assert set(ds.coords) == expected_coords

    # Check dimensions.
    assert set(ds.dims) == {"alleles", "ploidy", "samples", "variants"}

    # Check dim lengths.
    pos = api.snp_sites(region=region, field="POS", site_mask=site_mask)
    n_variants = len(pos)
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    n_samples = len(df_samples)
    assert ds.dims["variants"] == n_variants
    assert ds.dims["samples"] == n_samples
    assert ds.dims["ploidy"] == 2
    assert ds.dims["alleles"] == 4

    # Check shapes.
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xr.DataArray)
        assert isinstance(x.data, da.Array)

        if f == "variant_allele":
            assert x.ndim == 2
            assert x.shape == (n_variants, 4)
            assert x.dims == ("variants", "alleles")
        elif f.startswith("variant_"):
            assert x.ndim == 1
            assert x.shape == (n_variants,)
            assert x.dims == ("variants",)
        elif f in {"call_genotype", "call_genotype_mask"}:
            assert x.ndim == 3
            assert x.dims == ("variants", "samples", "ploidy")
            assert x.shape == (n_variants, n_samples, 2)
        elif f == "call_AD":
            assert x.ndim == 3
            assert x.dims == ("variants", "samples", "alleles")
            assert x.shape == (n_variants, n_samples, 4)
        elif f.startswith("call_"):
            assert x.ndim == 2
            assert x.dims == ("variants", "samples")
            assert x.shape == (n_variants, n_samples)
        elif f.startswith("sample_"):
            assert x.ndim == 1
            assert x.dims == ("samples",)
            assert x.shape == (n_samples,)

    # Check samples.
    expected_samples = df_samples["sample_id"].tolist()
    assert ds["sample_id"].values.tolist() == expected_samples

    # Check attributes.
    assert "contigs" in ds.attrs
    assert ds.attrs["contigs"] == api.contigs

    # Check can set up computations.
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xr.DataArray)
    d2 = ds["call_AD"].sum(axis=(1, 2))
    assert isinstance(d2, xr.DataArray)

    # Check compress bug.
    pos = ds["variant_position"].data
    assert pos.shape == pos.compute().shape


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_with_sample_sets_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    region = fixture.random_region_str()
    site_mask = random.choice((None,) + api.site_mask_ids)

    # Parametrize sample_sets.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_releases = api.releases
    parametrize_sample_sets = [
        None,
        random.choice(all_sample_sets),
        random.sample(all_sample_sets, 2),
        random.choice(all_releases),
    ]

    # Run tests.
    for sample_sets in parametrize_sample_sets:
        check_snp_calls(
            api=api, sample_sets=sample_sets, region=region, site_mask=site_mask
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_with_region_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice((None,) + api.site_mask_ids)

    # Parametrize region.
    contig = fixture.random_contig()
    df_gff = api.genome_features(attributes=["ID"])
    parametrize_region = [
        contig,
        fixture.random_region_str(),
        [fixture.random_region_str(), fixture.random_region_str()],
        random.choice(df_gff["ID"].dropna().to_list()),
    ]

    # Run tests.
    for region in parametrize_region:
        check_snp_calls(
            api=api, sample_sets=sample_sets, region=region, site_mask=site_mask
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_with_site_mask_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Parametrize site_mask.
    parametrize_site_mask = (None,) + api.site_mask_ids

    # Run tests.
    for site_mask in parametrize_site_mask:
        check_snp_calls(
            api=api, sample_sets=sample_sets, region=region, site_mask=site_mask
        )


@pytest.mark.parametrize(
    "sample_query",
    ["sex_call == 'F'", "taxon == 'coluzzii'", "taxon == 'robot'"],
)
def test_snp_calls_with_sample_query_param(ag3_sim_api: AnophelesSnpData, sample_query):
    df_samples = ag3_sim_api.sample_metadata().query(sample_query)

    if len(df_samples) == 0:
        with pytest.raises(ValueError):
            ag3_sim_api.snp_calls(region="3L", sample_query=sample_query)

    else:
        ds = ag3_sim_api.snp_calls(region="3L", sample_query=sample_query)
        assert ds.dims["samples"] == len(df_samples)
        assert_array_equal(ds["sample_id"].values, df_samples["sample_id"].values)


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_with_min_cohort_size_param(fixture, api: AnophelesSnpData):
    # Randomly fix some input parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Test with minimum cohort size.
    ds = api.snp_calls(
        sample_sets=sample_sets,
        region=region,
        min_cohort_size=10,
    )
    assert isinstance(ds, xr.Dataset)
    assert ds.dims["samples"] >= 10
    with pytest.raises(ValueError):
        api.snp_calls(
            sample_sets=sample_sets,
            region=region,
            min_cohort_size=1_000,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_with_max_cohort_size_param(fixture, api: AnophelesSnpData):
    # Randomly fix some input parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Test with maximum cohort size.
    ds = api.snp_calls(
        sample_sets=sample_sets,
        region=region,
        max_cohort_size=15,
    )
    assert isinstance(ds, xr.Dataset)
    assert ds.dims["samples"] <= 15


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_with_cohort_size_param(fixture, api: AnophelesSnpData):
    # Randomly fix some input parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Test with specific cohort size.
    cohort_size = random.randint(1, 10)
    ds = api.snp_calls(
        sample_sets=sample_sets,
        region=region,
        cohort_size=cohort_size,
    )
    assert isinstance(ds, xr.Dataset)
    assert ds.dims["samples"] == cohort_size
    with pytest.raises(ValueError):
        api.snp_calls(
            sample_sets=sample_sets,
            region=region,
            cohort_size=1_000,
        )


@pytest.mark.parametrize(
    "site_class",
    [
        "CDS_DEG_4",
        "CDS_DEG_2_SIMPLE",
        "CDS_DEG_0",
        "INTRON_SHORT",
        "INTRON_LONG",
        "INTRON_SPLICE_5PRIME",
        "INTRON_SPLICE_3PRIME",
        "UTR_5PRIME",
        "UTR_3PRIME",
        "INTERGENIC",
    ],
)
def test_snp_calls_with_site_class_param(ag3_sim_api: AnophelesSnpData, site_class):
    ds1 = ag3_sim_api.snp_calls(region="3L")
    ds2 = ag3_sim_api.snp_calls(region="3L", site_class=site_class)
    assert ds2.dims["variants"] < ds1.dims["variants"]


def check_snp_allele_counts(api, region, sample_sets, sample_query, site_mask):
    df_samples = api.sample_metadata(sample_sets=sample_sets, sample_query=sample_query)
    n_samples = len(df_samples)

    # Run once to compute results.
    ac = api.snp_allele_counts(
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        site_mask=site_mask,
    )
    assert isinstance(ac, np.ndarray)
    pos = api.snp_sites(region=region, field="POS", site_mask=site_mask)
    assert ac.shape == (pos.shape[0], 4)
    assert np.all(ac >= 0)
    an = ac.sum(axis=1)
    assert an.max() <= 2 * n_samples

    # Run again to ensure loading from results cache produces the same result.
    ac2 = api.snp_allele_counts(
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        site_mask=site_mask,
    )
    assert_array_equal(ac, ac2)


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_allele_counts_with_sample_sets_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    region = fixture.random_region_str()
    site_mask = random.choice((None,) + api.site_mask_ids)

    # Parametrize sample_sets.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_releases = api.releases
    parametrize_sample_sets = [
        None,
        random.choice(all_sample_sets),
        random.sample(all_sample_sets, 2),
        random.choice(all_releases),
    ]

    # Run tests.
    for sample_sets in parametrize_sample_sets:
        check_snp_allele_counts(
            api=api,
            sample_sets=sample_sets,
            region=region,
            site_mask=site_mask,
            sample_query=None,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_allele_counts_with_region_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice((None,) + api.site_mask_ids)

    # Parametrize region.
    contig = fixture.random_contig()
    df_gff = api.genome_features(attributes=["ID"])
    parametrize_region = [
        contig,
        fixture.random_region_str(),
        [fixture.random_region_str(), fixture.random_region_str()],
        random.choice(df_gff["ID"].dropna().to_list()),
    ]

    # Run tests.
    for region in parametrize_region:
        check_snp_allele_counts(
            api=api,
            sample_sets=sample_sets,
            region=region,
            site_mask=site_mask,
            sample_query=None,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_allele_counts_with_site_mask_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Parametrize site_mask.
    parametrize_site_mask = (None,) + api.site_mask_ids

    # Run tests.
    for site_mask in parametrize_site_mask:
        check_snp_allele_counts(
            api=api,
            sample_sets=sample_sets,
            region=region,
            site_mask=site_mask,
            sample_query=None,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_allele_counts_with_sample_query_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()
    site_mask = random.choice((None,) + api.site_mask_ids)

    # Parametrize sample_query.
    parametrize_sample_query = (None, "sex_call == 'F'")

    # Run tests.
    for sample_query in parametrize_sample_query:
        check_snp_allele_counts(
            api=api,
            sample_sets=sample_sets,
            region=region,
            site_mask=site_mask,
            sample_query=sample_query,
        )


def _check_is_accessible(api: AnophelesSnpData, region, mask):
    is_accessible = api.is_accessible(region=region, site_mask=mask)
    assert isinstance(is_accessible, np.ndarray)
    assert is_accessible.ndim == 1
    assert is_accessible.shape[0] == api.genome_sequence(region=region).shape[0]


@parametrize_with_cases("fixture,api", cases=".")
def test_is_accessible(fixture, api: AnophelesSnpData):
    # Parametrize region.
    contig = fixture.random_contig()
    df_gff = api.genome_features(attributes=["ID"])
    # Don't need to support multiple regions at this time.
    parametrize_region = [
        contig,
        fixture.random_region_str(),
        random.choice(df_gff["ID"].dropna().to_list()),
    ]

    # Parametrize site_mask.
    parametrize_site_mask = api.site_mask_ids

    # Run tests.
    for region, site_mask in product(
        parametrize_region,
        parametrize_site_mask,
    ):
        _check_is_accessible(
            api=api,
            region=region,
            mask=site_mask,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_snps(fixture, api: AnophelesSnpData):
    # Randomly choose parameter values.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()
    site_mask = random.choice(api.site_mask_ids)

    # Exercise the function.
    fig = api.plot_snps(
        region=region,
        sample_sets=sample_sets,
        site_mask=site_mask,
        show=False,
    )
    assert isinstance(fig, bokeh.model.Model)


def check_biallelic_snp_calls(api, sample_sets, region, site_mask):
    ds = api.biallelic_snp_calls(
        region=region, sample_sets=sample_sets, site_mask=site_mask
    )
    assert isinstance(ds, xr.Dataset)

    # Check fields.
    expected_data_vars = {
        "variant_allele",
        "variant_allele_count",
        "call_genotype",
    }
    assert set(ds.data_vars) == expected_data_vars

    expected_coords = {
        "variant_contig",
        "variant_position",
        "sample_id",
    }
    assert set(ds.coords) == expected_coords

    # Check dimensions.
    assert set(ds.dims) == {"alleles", "ploidy", "samples", "variants"}

    # Check dim lengths.
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    n_samples = len(df_samples)
    n_variants = ds.dims["variants"]
    assert ds.dims["samples"] == n_samples
    assert ds.dims["ploidy"] == 2
    assert ds.dims["alleles"] == 2

    # Check shapes.
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xr.DataArray)
        if f == "variant_allele_count":
            # This will have been computed.
            assert isinstance(x.data, np.ndarray)
        else:
            assert isinstance(x.data, da.Array)

        if f.startswith("variant_allele"):
            assert x.ndim == 2
            assert x.shape == (n_variants, 2)
            assert x.dims == ("variants", "alleles")
        elif f.startswith("variant_"):
            assert x.ndim == 1
            assert x.shape == (n_variants,)
            assert x.dims == ("variants",)
        elif f == "call_genotype":
            assert x.ndim == 3
            assert x.dims == ("variants", "samples", "ploidy")
            assert x.shape == (n_variants, n_samples, 2)
        elif f.startswith("sample_"):
            assert x.ndim == 1
            assert x.dims == ("samples",)
            assert x.shape == (n_samples,)

    # Check samples.
    expected_samples = df_samples["sample_id"].tolist()
    assert ds["sample_id"].values.tolist() == expected_samples

    # Check attributes.
    assert "contigs" in ds.attrs
    assert ds.attrs["contigs"] == api.contigs

    # Check can set up computations.
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xr.DataArray)

    # Check biallelic.
    gt = ds["call_genotype"].data
    assert gt.max().compute() <= 1

    # Check compress bug.
    pos = ds["variant_position"].data
    assert pos.shape == pos.compute().shape


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snp_calls_with_sample_sets_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    region = fixture.random_region_str()
    site_mask = random.choice((None,) + api.site_mask_ids)

    # Parametrize sample_sets.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_releases = api.releases
    parametrize_sample_sets = [
        None,
        random.choice(all_sample_sets),
        random.sample(all_sample_sets, 2),
        random.choice(all_releases),
    ]

    # Run tests.
    for sample_sets in parametrize_sample_sets:
        check_biallelic_snp_calls(
            api=api, sample_sets=sample_sets, region=region, site_mask=site_mask
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snp_calls_with_region_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice((None,) + api.site_mask_ids)

    # Parametrize region.
    contig = fixture.random_contig()
    df_gff = api.genome_features(attributes=["ID"])
    parametrize_region = [
        contig,
        fixture.random_region_str(),
        [fixture.random_region_str(), fixture.random_region_str()],
        random.choice(df_gff["ID"].dropna().to_list()),
    ]

    # Run tests.
    for region in parametrize_region:
        check_biallelic_snp_calls(
            api=api, sample_sets=sample_sets, region=region, site_mask=site_mask
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snp_calls_with_site_mask_param(fixture, api: AnophelesSnpData):
    # Fixed parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Parametrize site_mask.
    parametrize_site_mask = (None,) + api.site_mask_ids

    # Run tests.
    for site_mask in parametrize_site_mask:
        check_biallelic_snp_calls(
            api=api, sample_sets=sample_sets, region=region, site_mask=site_mask
        )


@pytest.mark.parametrize(
    "sample_query",
    ["sex_call == 'F'", "taxon == 'coluzzii'", "taxon == 'robot'"],
)
def test_biallelic_snp_calls_with_sample_query_param(
    ag3_sim_api: AnophelesSnpData, sample_query
):
    df_samples = ag3_sim_api.sample_metadata().query(sample_query)

    if len(df_samples) == 0:
        with pytest.raises(ValueError):
            ag3_sim_api.biallelic_snp_calls(region="3L", sample_query=sample_query)

    else:
        ds = ag3_sim_api.biallelic_snp_calls(region="3L", sample_query=sample_query)
        assert ds.dims["samples"] == len(df_samples)
        assert_array_equal(ds["sample_id"].values, df_samples["sample_id"].values)


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snp_calls_with_min_cohort_size_param(fixture, api: AnophelesSnpData):
    # Randomly fix some input parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Test with minimum cohort size.
    ds = api.biallelic_snp_calls(
        sample_sets=sample_sets,
        region=region,
        min_cohort_size=10,
    )
    assert isinstance(ds, xr.Dataset)
    assert ds.dims["samples"] >= 10
    with pytest.raises(ValueError):
        api.biallelic_snp_calls(
            sample_sets=sample_sets,
            region=region,
            min_cohort_size=1_000,
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snp_calls_with_max_cohort_size_param(fixture, api: AnophelesSnpData):
    # Randomly fix some input parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Test with maximum cohort size.
    ds = api.biallelic_snp_calls(
        sample_sets=sample_sets,
        region=region,
        max_cohort_size=15,
    )
    assert isinstance(ds, xr.Dataset)
    assert ds.dims["samples"] <= 15


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snp_calls_with_cohort_size_param(fixture, api: AnophelesSnpData):
    # Randomly fix some input parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    region = fixture.random_region_str()

    # Test with specific cohort size.
    cohort_size = random.randint(1, 10)
    ds = api.biallelic_snp_calls(
        sample_sets=sample_sets,
        region=region,
        cohort_size=cohort_size,
    )
    assert isinstance(ds, xr.Dataset)
    assert ds.dims["samples"] == cohort_size
    with pytest.raises(ValueError):
        api.biallelic_snp_calls(
            sample_sets=sample_sets,
            region=region,
            cohort_size=1_000,
        )


@pytest.mark.parametrize(
    "site_class",
    [
        "CDS_DEG_4",
        "CDS_DEG_2_SIMPLE",
        "CDS_DEG_0",
        "INTRON_SHORT",
        "INTRON_LONG",
        "INTRON_SPLICE_5PRIME",
        "INTRON_SPLICE_3PRIME",
        "UTR_5PRIME",
        "UTR_3PRIME",
        "INTERGENIC",
    ],
)
def test_biallelic_snp_calls_with_site_class_param(
    ag3_sim_api: AnophelesSnpData, site_class
):
    ds1 = ag3_sim_api.biallelic_snp_calls(region="3L")
    ds2 = ag3_sim_api.biallelic_snp_calls(region="3L", site_class=site_class)
    assert ds2.dims["variants"] < ds1.dims["variants"]
