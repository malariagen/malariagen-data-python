import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

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


def test_repr():
    af1 = setup_af1(check_location=True)
    assert isinstance(af1, Af1)
    r = repr(af1)
    assert isinstance(r, str)


# noinspection PyArgumentList
@pytest.mark.parametrize(
    "sample_sets",
    [
        "1229-VO-GH-DADZIE-VMF00095",
        ("1230-VO-GA-CF-AYALA-VMF00045", "1231-VO-MULTI-WONDJI-VMF00043"),
        "1.0",
    ],
)
@pytest.mark.parametrize("region", ["2RL", ["3RL", "X"], "3RL:28,000,000-29,000,000"])
def test_gene_cnv(region, sample_sets):
    af1 = setup_af1()

    ds = af1.gene_cnv(
        region=region, sample_sets=sample_sets, max_coverage_variance=None
    )

    assert isinstance(ds, xr.Dataset)

    # check fields
    expected_data_vars = {
        "CN_mode",
        "CN_mode_count",
        "gene_windows",
        "gene_contig",
        "gene_start",
        "gene_end",
        "gene_name",
        "gene_description",
        "gene_strand",
        "sample_coverage_variance",
        "sample_is_high_variance",
    }
    assert set(ds.data_vars) == expected_data_vars

    expected_coords = {
        "gene_id",
        "sample_id",
    }
    assert set(ds.coords) == expected_coords

    # check dimensions
    assert set(ds.dims) == {"samples", "genes"}

    # check dim lengths
    df_samples = af1.sample_metadata(sample_sets=sample_sets)
    n_samples = len(df_samples)
    assert ds.sizes["samples"] == n_samples
    df_genome_features = af1.genome_features(region=region)
    df_genes = df_genome_features.query("type == 'protein_coding_gene'")
    n_genes = len(df_genes)
    assert ds.sizes["genes"] == n_genes

    # check IDs
    assert ds["sample_id"].values.tolist() == df_samples["sample_id"].tolist()
    assert ds["gene_id"].values.tolist() == df_genes["ID"].tolist()

    # check shapes
    for f in expected_coords | expected_data_vars:
        x = ds[f]
        assert isinstance(x, xr.DataArray)
        assert isinstance(x.data, np.ndarray)

        if f.startswith("gene_"):
            assert x.ndim == 1
            assert x.dims == ("genes",)
        elif f.startswith("CN"):
            assert x.ndim == 2
            assert x.dims == ("genes", "samples")
        elif f.startswith("sample_"):
            assert x.ndim == 1
            assert x.dims == ("samples",)
            assert x.shape == (n_samples,)

    # check can set up computations
    d1 = ds["gene_start"] > 10_000
    assert isinstance(d1, xr.DataArray)
    d2 = ds["CN_mode"].max(axis=1)
    assert isinstance(d2, xr.DataArray)

    # sanity checks
    x = ds["gene_windows"].values
    y = ds["CN_mode_count"].values.max(axis=1)
    assert np.all(x >= y)
    z = ds["CN_mode"].values
    assert np.max(z) <= 12
    assert np.min(z) >= -1


@pytest.mark.parametrize(
    "region_raw",
    [
        "LOC125762289",
        "X",
        "2RL:48714463-48715355",
        "2RL:24,630,355-24,633,221",
        Region("2RL", 48714463, 48715355),
    ],
)
def test_locate_region(region_raw):
    # TODO Migrate this test.
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
    if region_raw == "X":
        assert region.contig == "X"
        assert region.start is None
        assert region.end is None

    # check that Region goes through unchanged
    if isinstance(region_raw, Region):
        assert region == region_raw

    # check that gene name matches coordinates from the geneset and matches gene sequence
    if region_raw == "LOC125762289":
        gene = gene_annotation.query("ID == 'LOC125762289'").squeeze()
        assert region == Region(gene.contig, gene.start, gene.end)
        assert pos[loc_region][0] == gene.start
        assert pos[loc_region][-1] == gene.end
        assert (
            ref[loc_region][:5].compute()
            == np.array(["T", "T", "T", "C", "T"], dtype="S1")
        ).all()

    # check string parsing
    if region_raw == "2RL:48714463-48715355":
        assert region == Region("2RL", 48714463, 48715355)
    if region_raw == "2RL:24,630,355-24,633,221":
        assert region == Region("2RL", 24630355, 24633221)


def test_h12_gwss():
    af1 = setup_af1(cohorts_analysis="20230823")
    sample_query = "country == 'Ghana'"
    contig = "3RL"
    analysis = "funestus"
    sample_sets = "1.0"
    window_size = 1000

    x, h12 = af1.h12_gwss(
        contig=contig,
        analysis=analysis,
        sample_query=sample_query,
        sample_sets=sample_sets,
        window_size=window_size,
        cohort_size=20,
    )

    # check dataset
    assert isinstance(x, np.ndarray)
    assert isinstance(h12, np.ndarray)

    # check dimensions
    assert len(x) == 15845
    assert len(x) == len(h12)

    # check some values
    assert_allclose(x[0], 185756.747)
    assert_allclose(h12[11353], 0.0525)


def test_h1x_gwss():
    af1 = setup_af1(cohorts_analysis="20230823")
    cohort1_query = "cohort_admin2_year == 'GH-NP_Kumbungu_fune_2017'"
    cohort2_query = "cohort_admin2_year == 'GH-NP_Zabzugu_fune_2017'"
    contig = "2RL"
    window_size = 2000

    x, h1x = af1.h1x_gwss(
        contig=contig,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        window_size=window_size,
        cohort_size=None,
        min_cohort_size=None,
        max_cohort_size=None,
    )

    # check data
    assert isinstance(x, np.ndarray)
    assert isinstance(h1x, np.ndarray)

    # check dimensions
    assert x.ndim == h1x.ndim == 1
    assert x.shape == h1x.shape

    # check some values
    assert_allclose(x[0], 87606.705, rtol=1e-5), x[0]
    assert_allclose(h1x[0], 0.008621, atol=1e-5), h1x[0]
    assert np.all(h1x <= 1)
    assert np.all(h1x >= 0)


def test_fst_gwss():
    af1 = setup_af1(cohorts_analysis="20230823")
    cohort1_query = "cohort_admin2_year == 'GH-NP_Kumbungu_fune_2017'"
    cohort2_query = "cohort_admin2_year == 'GH-NP_Zabzugu_fune_2017'"
    contig = "2RL"
    window_size = 10_000

    x, fst = af1.fst_gwss(
        contig=contig,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        window_size=window_size,
        cohort_size=None,
        min_cohort_size=None,
        max_cohort_size=None,
    )

    # check data
    assert isinstance(x, np.ndarray)
    assert isinstance(fst, np.ndarray)

    # check dimensions
    assert x.ndim == fst.ndim == 1
    assert x.shape == fst.shape

    # check some values
    assert_allclose(x[0], 87935.3098, rtol=1e-5), x[0]
    assert_allclose(fst[0], -0.105572, rtol=1e-5), fst[0]
    assert np.all(fst <= 1), np.max(fst)
    assert np.all(np.logical_and(fst >= -0.5, fst <= 1)), (np.min(fst), np.max(fst))
