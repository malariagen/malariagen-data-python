import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pandas.testing import assert_frame_equal

from malariagen_data import Af1, Region
from malariagen_data.util import locate_region, resolve_region


def setup_af1(url="simplecache::gs://vo_afun_release/", **kwargs):
    kwargs.setdefault("check_location", False)
    kwargs.setdefault("show_progress", False)
    if url is None:
        # test default URL
        # This only tests the setup_af1 default url, not the Af1 default.
        # The test_anopheles setup_subclass tests true defaults.
        return Af1(**kwargs)
    if url.startswith("simplecache::"):
        # configure the directory on the local file system to cache data
        kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Af1(url, **kwargs)


def test_sample_metadata():

    af1 = setup_af1()
    df_sample_sets_v1 = af1.sample_sets(release="1.0")

    expected_cols = (
        "sample_id",
        "partner_sample_id",
        "contributor",
        "country",
        "location",
        "year",
        "month",
        "latitude",
        "longitude",
        "sex_call",
        "sample_set",
        "release",
    )

    # all v1.0
    df_samples_v1 = af1.sample_metadata(sample_sets="1.0")
    assert tuple(df_samples_v1.columns[: len(expected_cols)]) == expected_cols
    expected_len = df_sample_sets_v1["sample_count"].sum()
    assert len(df_samples_v1) == expected_len

    # single sample set
    df_samples_eg = af1.sample_metadata(sample_sets="1229-VO-GH-DADZIE-VMF00095")
    assert tuple(df_samples_eg.columns[: len(expected_cols)]) == expected_cols
    expected_len = df_sample_sets_v1.query(
        "sample_set == '1229-VO-GH-DADZIE-VMF00095'"
    )["sample_count"].sum()
    assert len(df_samples_eg) == expected_len

    # multiple sample sets
    sample_sets = [
        "1229-VO-GH-DADZIE-VMF00095",
        "1230-VO-GA-CF-AYALA-VMF00045",
        "1231-VO-MULTI-WONDJI-VMF00043",
    ]
    df_samples_egs = af1.sample_metadata(sample_sets=sample_sets)
    assert tuple(df_samples_egs.columns[: len(expected_cols)]) == expected_cols
    loc_sample_sets = df_sample_sets_v1["sample_set"].isin(sample_sets)
    expected_len = df_sample_sets_v1.loc[loc_sample_sets]["sample_count"].sum()
    assert len(df_samples_egs) == expected_len

    # duplicate sample sets
    with pytest.raises(ValueError):
        af1.sample_metadata(sample_sets=["1.0", "1.0"])
    with pytest.raises(ValueError):
        af1.sample_metadata(
            sample_sets=["1229-VO-GH-DADZIE-VMF00095", "1229-VO-GH-DADZIE-VMF00095"]
        )
    with pytest.raises(ValueError):
        af1.sample_metadata(sample_sets=["1229-VO-GH-DADZIE-VMF00095", "1.0"])

    # default is all public releases
    df_default = af1.sample_metadata()
    df_all = af1.sample_metadata(sample_sets=af1.releases)
    assert_frame_equal(df_default, df_all)


@pytest.mark.parametrize(
    "region",
    ["2RL", ["2RL", "3RL", "2RL:48,714,463-48,715,355", "gene-LOC125762289"]],
)
def test_site_filters(region):
    af1 = setup_af1()
    filter_pass = af1.site_filters(region=region, mask="funestus")
    assert isinstance(filter_pass, da.Array)
    assert filter_pass.ndim == 1
    assert filter_pass.dtype == bool


@pytest.mark.parametrize("chunks", ["auto", "native"])
@pytest.mark.parametrize(
    "region",
    ["2RL", ["3RL", "2RL:48,714,463-48,715,355", "gene-LOC125762289"]],
)
def test_snp_sites(chunks, region):

    af1 = setup_af1()

    pos = af1.snp_sites(region=region, field="POS", chunks=chunks)
    ref = af1.snp_sites(region=region, field="REF", chunks=chunks)
    alt = af1.snp_sites(region=region, field="ALT", chunks=chunks)
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

    # apply site mask
    filter_pass = af1.site_filters(region=region, mask="funestus").compute()
    n_pass = np.count_nonzero(filter_pass)
    pos_pass = af1.snp_sites(
        region=region, field="POS", site_mask="funestus", chunks=chunks
    )
    assert isinstance(pos_pass, da.Array)
    assert pos_pass.ndim == 1
    assert pos_pass.dtype == "i4"
    assert pos_pass.shape[0] == n_pass
    assert pos_pass.compute().shape == pos_pass.shape
    for f in "POS", "REF", "ALT":
        d = af1.snp_sites(region=region, site_mask="funestus", field=f, chunks=chunks)
        assert isinstance(d, da.Array)
        assert d.shape[0] == n_pass
        assert d.shape == d.compute().shape


@pytest.mark.parametrize("chunks", ["auto", "native"])
@pytest.mark.parametrize(
    "sample_sets",
    [
        None,
        "1229-VO-GH-DADZIE-VMF00095",
        ["1240-VO-CD-KOEKEMOER-VMF00099", "1240-VO-MZ-KOEKEMOER-VMF00101"],
        "1.0",
    ],
)
@pytest.mark.parametrize(
    "region",
    ["2RL", ["3RL", "2RL:48,714,463-48,715,355", "gene-LOC125762289"]],
)
def test_snp_genotypes(chunks, sample_sets, region):

    af1 = setup_af1()

    df_samples = af1.sample_metadata(sample_sets=sample_sets)
    gt = af1.snp_genotypes(region=region, sample_sets=sample_sets, chunks=chunks)
    assert isinstance(gt, da.Array)
    assert gt.ndim == 3
    assert gt.dtype == "i1"
    assert gt.shape[1] == len(df_samples)

    # specific fields
    x = af1.snp_genotypes(
        region=region, sample_sets=sample_sets, field="GT", chunks=chunks
    )
    assert isinstance(x, da.Array)
    assert x.ndim == 3
    assert x.dtype == "i1"

    x = af1.snp_genotypes(
        region=region, sample_sets=sample_sets, field="GQ", chunks=chunks
    )
    assert isinstance(x, da.Array)
    assert x.ndim == 2
    # FIXME: test_ag3.py asserts this instead, which only passes for Ag3.0 (not Ag3.x):
    # assert x.dtype == "i2"
    assert x.dtype == "int8"

    x = af1.snp_genotypes(
        region=region, sample_sets=sample_sets, field="MQ", chunks=chunks
    )
    assert isinstance(x, da.Array)
    assert x.ndim == 2
    # FIXME: test_ag3.py asserts this instead, which only passes for Ag3.0 (not Ag3.x)
    # assert x.dtype == "i2"
    assert x.dtype == "float32"

    x = af1.snp_genotypes(
        region=region, sample_sets=sample_sets, field="AD", chunks=chunks
    )
    assert isinstance(x, da.Array)
    assert x.ndim == 3
    assert x.dtype == "i2"

    # site mask
    filter_pass = af1.site_filters(region=region, mask="funestus").compute()
    gt_pass = af1.snp_genotypes(
        region=region,
        sample_sets=sample_sets,
        site_mask="funestus",
        chunks=chunks,
    )
    assert isinstance(gt_pass, da.Array)
    assert gt_pass.ndim == 3
    assert gt_pass.dtype == "i1"
    assert gt_pass.shape[0] == np.count_nonzero(filter_pass)
    assert gt_pass.shape[1] == len(df_samples)
    assert gt_pass.shape[2] == 2


@pytest.mark.parametrize(
    "sample_sets",
    [
        None,
        "1229-VO-GH-DADZIE-VMF00095",
        ["1230-VO-GA-CF-AYALA-VMF00045", "1231-VO-MULTI-WONDJI-VMF00043"],
        "1.0",
    ],
)
@pytest.mark.parametrize(
    "region",
    ["2RL", ["3RL", "2RL:48,714,463-48,715,355", "gene-LOC125762289"]],
)
def test_snp_genotypes_chunks(sample_sets, region):

    af1 = setup_af1()
    gt_native = af1.snp_genotypes(
        region=region, sample_sets=sample_sets, chunks="native"
    )
    gt_auto = af1.snp_genotypes(region=region, sample_sets=sample_sets, chunks="auto")
    gt_manual = af1.snp_genotypes(
        region=region, sample_sets=sample_sets, chunks=(100_000, 10, 2)
    )

    assert gt_native.chunks != gt_auto.chunks
    assert gt_auto.chunks != gt_manual.chunks
    assert gt_manual.chunks != gt_native.chunks
    assert gt_manual.chunks[0][0] == 100_000
    assert gt_manual.chunks[1][0] == 10
    assert gt_manual.chunks[2][0] == 2


@pytest.mark.parametrize(
    "sample_sets",
    [
        None,
        "1229-VO-GH-DADZIE-VMF00095",
        ["1240-VO-CD-KOEKEMOER-VMF00099", "1240-VO-MZ-KOEKEMOER-VMF00101"],
        "1.0",
    ],
)
@pytest.mark.parametrize(
    "region",
    ["2RL", ["3RL", "2RL:48,714,463-48,715,355", "gene-LOC125762289"]],
)
@pytest.mark.parametrize(
    "site_mask",
    [None, "funestus"],
)
def test_snp_calls(sample_sets, region, site_mask):

    af1 = setup_af1()

    ds = af1.snp_calls(region=region, sample_sets=sample_sets, site_mask=site_mask)
    assert isinstance(ds, xr.Dataset)

    # check fields
    expected_data_vars = {
        "variant_allele",
        "variant_filter_pass_funestus",
        "call_genotype",
        "call_genotype_mask",
        "call_GQ",
        "call_AD",
        "call_MQ",
    }
    assert set(ds.data_vars) == expected_data_vars

    expected_coords = {
        "variant_contig",
        "variant_position",
        "sample_id",
    }
    assert set(ds.coords) == expected_coords

    # check dimensions
    assert set(ds.dims) == {"alleles", "ploidy", "samples", "variants"}

    # check dim lengths
    pos = af1.snp_sites(region=region, field="POS", site_mask=site_mask)
    n_variants = len(pos)
    df_samples = af1.sample_metadata(sample_sets=sample_sets)
    n_samples = len(df_samples)
    assert ds.dims["variants"] == n_variants
    assert ds.dims["samples"] == n_samples
    assert ds.dims["ploidy"] == 2
    assert ds.dims["alleles"] == 4

    # check shapes
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

    # check samples
    expected_samples = df_samples["sample_id"].tolist()
    assert ds["sample_id"].values.tolist() == expected_samples

    # check attributes
    assert "contigs" in ds.attrs
    assert ds.attrs["contigs"] == ("2RL", "3RL", "X")

    # check can set up computations
    d1 = ds["variant_position"] > 10_000
    assert isinstance(d1, xr.DataArray)
    d2 = ds["call_AD"].sum(axis=(1, 2))
    assert isinstance(d2, xr.DataArray)

    # check compress bug
    pos = ds["variant_position"].data
    assert pos.shape == pos.compute().shape


@pytest.mark.parametrize(
    "region",
    ["gene-LOC125762289", "2RL:48714463-48715355", "3RL", "X"],
)
def test_is_accessible(region):

    af1 = setup_af1()
    # run a couple of tests
    is_accessible = af1.is_accessible(region=region, site_mask="funestus")
    assert isinstance(is_accessible, np.ndarray)
    assert is_accessible.ndim == 1
    assert is_accessible.shape[0] == af1.genome_sequence(region).shape[0]


@pytest.mark.parametrize(
    "region_raw",
    [
        "gene-LOC125762289",
        "3RL",
        "2RL:48714463-48715355",
        "2RL:24,630,355-24,633,221",
        Region("2RL", 48714463, 48715355),
    ],
)
def test_locate_region(region_raw):

    af1 = setup_af1()
    gene_annotation = af1.geneset(attributes=["ID"])
    region = resolve_region(af1, region_raw)
    pos = af1.snp_sites(region=region.contig, field="POS")
    ref = af1.snp_sites(region=region.contig, field="REF")
    loc_region = locate_region(region, pos)

    # check types
    assert isinstance(loc_region, slice)
    assert isinstance(region, Region)

    # check Region with contig
    if region_raw == "3RL":
        assert region.contig == "3RL"
        assert region.start is None
        assert region.end is None

    # check that Region goes through unchanged
    if isinstance(region_raw, Region):
        assert region == region_raw

    # check that gene name matches coordinates from the geneset and matches gene sequence
    if region_raw == "gene-LOC125762289":
        gene = gene_annotation.query("ID == 'gene-LOC125762289'").squeeze()
        assert region == Region(gene.contig, gene.start, gene.end)
        assert pos[loc_region][0] == gene.start
        assert pos[loc_region][-1] == gene.end
        # TODO: check this is the expected sequence
        assert (
            ref[loc_region][:5].compute()
            == np.array(["T", "T", "T", "C", "T"], dtype="S1")
        ).all()

    # check string parsing
    if region_raw == "2RL:48714463-48715355":
        assert region == Region("2RL", 48714463, 48715355)
    if region_raw == "2RL:24,630,355-24,633,221":
        assert region == Region("2RL", 24630355, 24633221)


# FIXME: testing "2RL" via CI causes exit code 137 (out of memory)
@pytest.mark.parametrize("region", ["X", "gene-LOC125762289", "2RL:48714463-48715355"])
@pytest.mark.parametrize("site_mask", [None, "funestus"])
def test_site_annotations(region, site_mask):

    af1 = setup_af1()

    ds_snp = af1.snp_variants(region=region, site_mask=site_mask)
    n_variants = ds_snp.dims["variants"]
    ds_ann = af1.site_annotations(region=region, site_mask=site_mask)
    # site annotations dataset is aligned with SNP sites
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
