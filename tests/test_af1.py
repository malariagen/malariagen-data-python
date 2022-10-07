#

import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from pandas.testing import assert_frame_equal

from malariagen_data import Af1


def setup_af1(url="simplecache::gs://vo_afun_release/", **kwargs):
    kwargs.setdefault("check_location", False)
    kwargs.setdefault("show_progress", False)
    if url is None:
        # test default URL
        return Af1(**kwargs)
    if url.startswith("simplecache::"):
        # configure the directory on the local file system to cache data
        kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Af1(url, **kwargs)


@pytest.mark.parametrize(
    "url",
    [
        None,
        "gs://vo_afun_release/",
        "gcs://vo_afun_release/",
        "gs://vo_afun_release",
        "gcs://vo_afun_release",
        "simplecache::gs://vo_afun_release/",
        "simplecache::gcs://vo_afun_release/",
    ],
)
def test_sample_sets(url):

    af1 = setup_af1(url)
    df_sample_sets_v1 = af1.sample_sets(release="1.0")
    assert isinstance(df_sample_sets_v1, pd.DataFrame)
    assert len(df_sample_sets_v1) == 8
    assert tuple(df_sample_sets_v1.columns) == ("sample_set", "sample_count", "release")

    # test duplicates not allowed
    with pytest.raises(ValueError):
        af1.sample_sets(release=["1.0", "1.0"])

    # test default is all public releases
    df_default = af1.sample_sets()
    df_all = af1.sample_sets(release=af1.releases)
    assert_frame_equal(df_default, df_all)


def test_releases():

    af1 = setup_af1()
    assert isinstance(af1.releases, tuple)
    assert af1.releases == ("1.0",)

    af1 = setup_af1(pre=True)
    assert isinstance(af1.releases, tuple)
    # FIXME: test_ag3.py has assert len(ag3.releases) > 1
    assert len(af1.releases) > 0
    assert all([r.startswith("1.") for r in af1.releases])


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


@pytest.mark.parametrize("mask", ["funestus"])
def test_open_site_filters(mask):
    # check can open the zarr directly
    af1 = setup_af1()
    root = af1.open_site_filters(mask=mask)
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in af1.contigs:
        assert contig in root


@pytest.mark.parametrize("mask", ["funestus"])
@pytest.mark.parametrize(
    # FIXME: support region and gene names
    # "region", ["2RL", ["2RL", "3RL", "2RL:48,714,463-48,715,355", "AGAP007280"]]
    "region",
    ["2RL", ["2RL", "3RL"]],
)
def test_site_filters(mask, region):
    af1 = setup_af1()
    filter_pass = af1.site_filters(region=region, mask=mask)
    assert isinstance(filter_pass, da.Array)
    assert filter_pass.ndim == 1
    assert filter_pass.dtype == bool


def test_open_snp_sites():
    af1 = setup_af1()
    root = af1.open_snp_sites()
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in af1.contigs:
        assert contig in root


@pytest.mark.parametrize("chunks", ["auto", "native"])
@pytest.mark.parametrize(
    # FIXME: support region and gene names
    # "region", ["2RL", ["3RL", "2RL:48,714,463-48,715,355", "AGAP007280"]]
    "region",
    ["2RL", ["2RL", "3RL"]],
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


def test_open_snp_genotypes():
    # check can open the zarr directly
    af1 = setup_af1()
    root = af1.open_snp_genotypes(sample_set="1229-VO-GH-DADZIE-VMF00095")
    assert isinstance(root, zarr.hierarchy.Group)
    for contig in af1.contigs:
        assert contig in root


@pytest.mark.parametrize("chunks", ["auto", "native"])
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
    # FIXME: support region and gene names
    # "region", ["2RL", ["3RL", "2RL:48,714,463-48,715,355", "AGAP007280"]]
    "region",
    ["2RL", ["2RL", "3RL"]],
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
    # FIXME: test_ag3.py asserts this instead:
    # assert x.dtype == "i2"
    assert x.dtype == "int8"
    x = af1.snp_genotypes(
        region=region, sample_sets=sample_sets, field="MQ", chunks=chunks
    )
    assert isinstance(x, da.Array)
    assert x.ndim == 2
    # FIXME: test_ag3.py asserts this instead:
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
    # FIXME: support region and gene names
    # "region", ["2RL", ["3RL", "2RL:48,714,463-48,715,355", "AGAP007280"]]
    "region",
    ["2RL", ["2RL", "3RL"]],
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
