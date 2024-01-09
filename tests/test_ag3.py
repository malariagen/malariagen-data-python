import random

import numpy as np
import pandas as pd
import pytest
import scipy.stats  # type: ignore
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from pandas.testing import assert_frame_equal

from malariagen_data import Ag3, Region
from malariagen_data.anopheles import _cn_mode
from malariagen_data.util import locate_region, resolve_region, compare_series_like

contigs = "2R", "2L", "3R", "3L", "X"


def setup_ag3(url="simplecache::gs://vo_agam_release/", **kwargs):
    kwargs.setdefault("check_location", False)
    kwargs.setdefault("show_progress", False)
    if url is None:
        # test default URL
        # This only tests the setup_af1 default url, not the Ag3 default.
        # The test_anopheles setup_subclass tests true defaults.
        return Ag3(**kwargs)
    if url.startswith("simplecache::"):
        # configure the directory on the local file system to cache data
        kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Ag3(url, **kwargs)


def test_repr():
    ag3 = setup_ag3(check_location=True)
    assert isinstance(ag3, Ag3)
    r = repr(ag3)
    assert isinstance(r, str)


def test_cross_metadata():
    ag3 = setup_ag3()
    df_crosses = ag3.cross_metadata()
    assert isinstance(df_crosses, pd.DataFrame)
    expected_cols = ["cross", "sample_id", "father_id", "mother_id", "sex", "role"]
    assert df_crosses.columns.tolist() == expected_cols

    # check samples are in AG1000G-X
    df_samples = ag3.sample_metadata(sample_sets="AG1000G-X")
    assert set(df_crosses["sample_id"]) == set(df_samples["sample_id"])

    # check values
    expected_role_values = ["parent", "progeny"]
    assert df_crosses["role"].unique().tolist() == expected_role_values
    expected_sex_values = ["F", "M"]
    assert df_crosses["sex"].unique().tolist() == expected_sex_values


@pytest.mark.parametrize("rows", [10, 100, 1000])
@pytest.mark.parametrize("cols", [10, 100, 1000])
@pytest.mark.parametrize("vmax", [2, 12, 100])
def test_cn_mode(rows, cols, vmax):
    """Test the numba-optimised function for computing modal copy number."""

    a = np.random.randint(0, vmax, size=(rows * cols), dtype="i1").reshape(rows, cols)
    expect = scipy.stats.mode(a, axis=0)
    modes, counts = _cn_mode(a, vmax)
    assert_array_equal(expect.mode.squeeze(), modes)
    assert_array_equal(expect.count.squeeze(), counts)


# noinspection PyArgumentList
@pytest.mark.parametrize(
    "sample_sets",
    ["AG1000G-AO", ("AG1000G-TZ", "AG1000G-UG"), "3.0"],
)
@pytest.mark.parametrize(
    "region", ["2R", "X", ["2R", "3R"], "3R:28,000,000-29,000,000"]
)
def test_gene_cnv(region, sample_sets):
    ag3 = setup_ag3()

    ds = ag3.gene_cnv(
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
    df_samples = ag3.sample_metadata(sample_sets=sample_sets)
    n_samples = len(df_samples)
    assert ds.sizes["samples"] == n_samples
    df_genome_features = ag3.genome_features(region=region)
    df_genes = df_genome_features.query("type == 'gene'")
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
    "sample_sets",
    ["AG1000G-AO", ("AG1000G-TZ", "AG1000G-UG"), "3.0"],
)
@pytest.mark.parametrize("region", ["2R", "X", "3R:28,000,000-29,000,000"])
def test_gene_cnv_xarray_indexing(region, sample_sets):
    ag3 = setup_ag3()

    ds = ag3.gene_cnv(
        region=region, sample_sets=sample_sets, max_coverage_variance=None
    )

    # check label-based indexing
    # pick a random gene and sample ID

    # check dim lengths
    df_samples = ag3.sample_metadata(sample_sets=sample_sets)
    df_genome_features = ag3.genome_features(region=region)
    df_genes = df_genome_features.query("type == 'gene'")
    gene = random.choice(df_genes["ID"].tolist())
    sample = random.choice(df_samples["sample_id"].tolist())
    ds = ds.set_index(genes="gene_id", samples="sample_id")
    o = ds.sel(genes=gene)
    assert isinstance(o, xr.Dataset)
    assert set(o.dims) == {"samples"}
    assert o.sizes["samples"] == ds.sizes["samples"]
    o = ds.sel(samples=sample)
    assert isinstance(o, xr.Dataset)
    assert set(o.dims) == {"genes"}
    assert o.sizes["genes"] == ds.sizes["genes"]
    o = ds.sel(genes=gene, samples=sample)
    assert isinstance(o, xr.Dataset)
    assert set(o.dims) == set()


def _check_frequency(x):
    loc_nan = np.isnan(x)
    assert np.all(x[~loc_nan] >= 0)
    assert np.all(x[~loc_nan] <= 1)


@pytest.mark.parametrize(
    "region", ["2R", "X", ["2R", "3R"], "3R:28,000,000-29,000,000"]
)
@pytest.mark.parametrize(
    "cohorts",
    [
        {
            "ke": "country == 'Kenya'",
            "bf_2012_col": "country == 'Burkina Faso' and year == 2012 and aim_species == 'coluzzii'",
        },
        "admin1_month",
    ],
)
def test_gene_cnv_frequencies(region, cohorts):
    ag3 = setup_ag3(cohorts_analysis="20230516")

    universal_fields = [
        "contig",
        "start",
        "end",
        "windows",
        "max_af",
        "gene_strand",
        "gene_description",
        "label",
    ]
    df_genes = ag3.genome_features(region=region).query("type == 'gene'")

    df_cnv_frq = ag3.gene_cnv_frequencies(
        region=region,
        sample_sets="3.0",
        cohorts=cohorts,
        min_cohort_size=1,
        drop_invariant=False,
        max_coverage_variance=None,
    )

    assert isinstance(df_cnv_frq, pd.DataFrame)
    assert len(df_cnv_frq) == len(df_genes) * 2
    assert df_cnv_frq.index.names == ["gene_id", "gene_name", "cnv_type"]

    # sanity checks
    frq_cols = None
    if isinstance(cohorts, dict):
        frq_cols = ["frq_" + s for s in cohorts.keys()]
    if isinstance(cohorts, str):
        df_coh = ag3.cohorts_metadata(sample_sets="3.0")
        coh_nm = "cohort_" + cohorts
        frq_cols = ["frq_" + s for s in list(df_coh[coh_nm].dropna().unique())]

    # check frequencies are within sensible range
    for f in frq_cols:
        _check_frequency(df_cnv_frq[f].values)

    # check amp and del frequencies are within sensible range
    df_frq_amp = df_cnv_frq[frq_cols].xs("amp", level="cnv_type")
    df_frq_del = df_cnv_frq[frq_cols].xs("del", level="cnv_type")
    df_frq_sum = df_frq_amp + df_frq_del
    for f in frq_cols:
        _check_frequency(df_frq_sum[f].values)
    expected_fields = universal_fields + frq_cols
    assert sorted(df_cnv_frq.columns.tolist()) == sorted(expected_fields)


def test_gene_cnv_frequencies__query():
    ag3 = setup_ag3(cohorts_analysis="20230516")

    region = "3L"

    expected_columns = [
        "contig",
        "start",
        "end",
        "windows",
        "max_af",
        "gene_strand",
        "gene_description",
        "label",
        "frq_AO-LUA_colu_2009",
    ]

    df = ag3.gene_cnv_frequencies(
        region=region,
        sample_sets="3.0",
        cohorts="admin1_year",
        min_cohort_size=10,
        sample_query="country == 'Angola'",
        drop_invariant=False,
    )

    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns) == sorted(expected_columns)
    df_genes = ag3.genome_features(region=region).query("type == 'gene'")
    assert len(df) == len(df_genes) * 2


def test_gene_cnv_frequencies__max_coverage_variance():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    region = "3L"
    df_genes = ag3.genome_features(region=region).query("type == 'gene'")

    base_columns = [
        "contig",
        "start",
        "end",
        "windows",
        "max_af",
        "gene_strand",
        "gene_description",
        "label",
    ]

    # run without a threshold on coverage variance
    df = ag3.gene_cnv_frequencies(
        region=region,
        sample_sets=["AG1000G-GM-A", "AG1000G-GM-B", "AG1000G-GM-C"],
        cohorts="admin1_year",
        min_cohort_size=10,
        max_coverage_variance=None,
        drop_invariant=False,
    )
    expected_frq_columns = [
        "frq_GM-L_gcx2_2006",
        "frq_GM-M_gcx2_2012",
        "frq_GM-N_gcx1_2011",
    ]
    expected_columns = base_columns + expected_frq_columns
    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns) == sorted(expected_columns)
    assert len(df) == len(df_genes) * 2

    # Run with a threshold on coverage variance - this will remove samples,
    # which in turn will drop one of the cohorts below the min_cohort_size,
    # and so we can check that we have lost a cohort.
    df = ag3.gene_cnv_frequencies(
        region=region,
        sample_sets=["AG1000G-GM-A", "AG1000G-GM-B", "AG1000G-GM-C"],
        cohorts="admin1_year",
        min_cohort_size=10,
        max_coverage_variance=0.2,
        drop_invariant=False,
    )
    expected_frq_columns = [
        "frq_GM-M_gcx2_2012",
        "frq_GM-N_gcx1_2011",
    ]
    expected_columns = base_columns + expected_frq_columns
    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns) == sorted(expected_columns)
    assert len(df) == len(df_genes) * 2


def test_gene_cnv_frequencies__drop_invariant():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    region = "3L"

    expected_columns = [
        "contig",
        "start",
        "end",
        "windows",
        "max_af",
        "gene_strand",
        "gene_description",
        "label",
        "frq_AO-LUA_colu_2009",
    ]

    df = ag3.gene_cnv_frequencies(
        region=region,
        sample_sets="3.0",
        cohorts="admin1_year",
        min_cohort_size=10,
        sample_query="country == 'Angola'",
        drop_invariant=True,
    )

    assert isinstance(df, pd.DataFrame)
    assert sorted(df.columns) == sorted(expected_columns)
    assert np.all(df["max_af"] > 0)
    df_genes = ag3.genome_features(region=region).query("type == 'gene'")
    assert len(df) < len(df_genes) * 2


def test_gene_cnv_frequencies__dup_samples():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    df_dup = ag3.gene_cnv_frequencies(
        region="3L",
        cohorts="admin1_year",
        sample_sets=["AG1000G-FR", "AG1000G-FR"],
    )
    df = ag3.gene_cnv_frequencies(
        region="3L",
        cohorts="admin1_year",
        sample_sets=["AG1000G-FR"],
    )
    assert_frame_equal(df, df_dup)


def test_gene_cnv_frequencies__multi_contig_x():
    # https://github.com/malariagen/malariagen-data-python/issues/166

    ag3 = setup_ag3(cohorts_analysis="20230516")

    df1 = ag3.gene_cnv_frequencies(
        region="X",
        sample_sets="AG1000G-BF-B",
        cohorts="admin1_year",
        min_cohort_size=10,
        drop_invariant=False,
        max_coverage_variance=None,
    )

    df2 = ag3.gene_cnv_frequencies(
        region=["2R", "X"],
        sample_sets="AG1000G-BF-B",
        cohorts="admin1_year",
        min_cohort_size=10,
        drop_invariant=False,
        max_coverage_variance=None,
    ).query("contig == 'X'")

    assert_frame_equal(df1, df2)


def test_gene_cnv_frequencies__missing_samples():
    # https://github.com/malariagen/malariagen-data-python/issues/183

    ag3 = setup_ag3(cohorts_analysis="20230516", pre=True)

    df = ag3.gene_cnv_frequencies(
        region="3L",
        sample_sets="1190-VO-GH-AMENGA-ETEGO-VMF00013",
        cohorts="admin1_year",
    )
    assert isinstance(df, pd.DataFrame)


@pytest.mark.parametrize(
    "region_raw",
    [
        "AGAP007280",
        "3L",
        "2R:48714463-48715355",
        "2R:24,630,355-24,633,221",
        Region("2R", 48714463, 48715355),
    ],
)
def test_locate_region(region_raw):
    # TODO Migrate this test.
    ag3 = setup_ag3()
    gene_annotation = ag3.genome_features(attributes=["ID"])
    region = resolve_region(ag3, region_raw)
    pos = ag3.snp_sites(region=region.contig, field="POS")
    ref = ag3.snp_sites(region=region.contig, field="REF")
    loc_region = locate_region(region, pos)

    # check types
    assert isinstance(loc_region, slice)
    assert isinstance(region, Region)

    # check Region with contig
    if region_raw == "3L":
        assert region.contig == "3L"
        assert region.start is None
        assert region.end is None

    # check that Region goes through unchanged
    if isinstance(region_raw, Region):
        assert region == region_raw

    # check that gene name matches coordinates from the genome_features and matches gene sequence
    if region_raw == "AGAP007280":
        gene = gene_annotation.query("ID == 'AGAP007280'").squeeze()
        assert region == Region(gene.contig, gene.start, gene.end)
        assert pos[loc_region][0] == gene.start
        assert pos[loc_region][-1] == gene.end
        assert (
            ref[loc_region][:5].compute()
            == np.array(["A", "T", "G", "G", "C"], dtype="S1")
        ).all()

    # check string parsing
    if region_raw == "2R:48714463-48715355":
        assert region == Region("2R", 48714463, 48715355)
    if region_raw == "2R:24,630,355-24,633,221":
        assert region == Region("2R", 24630355, 24633221)


# noinspection PyDefaultArgument
def _check_gene_cnv_frequencies_advanced(
    region="2L",
    area_by="admin1_iso",
    period_by="year",
    sample_sets=["AG1000G-BF-A", "AG1000G-ML-A", "AG1000G-UG"],
    sample_query=None,
    min_cohort_size=10,
    variant_query="max_af > 0.02",
    drop_invariant=True,
    max_coverage_variance=0.2,
):
    ag3 = setup_ag3(cohorts_analysis="20230516")

    ds = ag3.gene_cnv_frequencies_advanced(
        region=region,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        sample_query=sample_query,
        min_cohort_size=min_cohort_size,
        variant_query=variant_query,
        drop_invariant=drop_invariant,
        max_coverage_variance=max_coverage_variance,
    )

    assert isinstance(ds, xr.Dataset)

    # noinspection PyTypeChecker
    assert sorted(ds.dims) == ["cohorts", "variants"]

    expected_variant_vars = (
        "variant_label",
        "variant_contig",
        "variant_start",
        "variant_end",
        "variant_windows",
        "variant_cnv_type",
        "variant_gene_id",
        "variant_gene_name",
        "variant_gene_strand",
        "variant_max_af",
    )
    for v in expected_variant_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants",)

    expected_cohort_vars = (
        "cohort_label",
        "cohort_size",
        "cohort_taxon",
        "cohort_area",
        "cohort_period",
        "cohort_period_start",
        "cohort_period_end",
        "cohort_lat_mean",
        "cohort_lat_min",
        "cohort_lat_max",
        "cohort_lon_mean",
        "cohort_lon_min",
        "cohort_lon_max",
    )
    for v in expected_cohort_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("cohorts",)

    expected_event_vars = (
        "event_count",
        "event_nobs",
        "event_frequency",
        "event_frequency_ci_low",
        "event_frequency_ci_upp",
    )
    for v in expected_event_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants", "cohorts")

    # sanity checks for area values
    df_samples = ag3.sample_metadata(sample_sets=sample_sets)
    if sample_query is not None:
        df_samples = df_samples.query(sample_query)
    expected_area = np.unique(df_samples[area_by].dropna().values)
    area = ds["cohort_area"].values
    # N.B., some areas may not end up in final dataset if cohort
    # size is too small, so do a set membership test
    for a in area:
        assert a in expected_area

    # sanity checks for period values
    period = ds["cohort_period"].values
    if period_by == "year":
        expected_freqstr = "A-DEC"
    elif period_by == "month":
        expected_freqstr = "M"
    elif period_by == "quarter":
        expected_freqstr = "Q-DEC"
    else:
        assert False, "not implemented"
    for p in period:
        assert isinstance(p, pd.Period)
        assert p.freqstr == expected_freqstr

    # sanity check cohort size
    size = ds["cohort_size"].values
    for s in size:
        assert s >= min_cohort_size

    if area_by == "admin1_iso" and period_by == "year":
        # Here we test the behaviour of the function when grouping by admin level
        # 1 and year. We can do some more in-depth testing in this case because
        # we can compare results directly against the simpler gene_cnv_frequencies()
        # function with the admin1_year cohorts.

        # check consistency with the basic gene CNV frequencies method
        df_af = ag3.gene_cnv_frequencies(
            region=region,
            cohorts="admin1_year",
            sample_sets=sample_sets,
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
            drop_invariant=drop_invariant,
            max_coverage_variance=max_coverage_variance,
        )
        df_af = df_af.reset_index()  # make sure all variables available to check
        if variant_query is not None:
            df_af = df_af.query(variant_query)

        # check cohorts are consistent
        expect_cohort_labels = sorted(
            [c.split("frq_")[1] for c in df_af.columns if c.startswith("frq_")]
        )
        cohort_labels = sorted(ds["cohort_label"].values)
        assert cohort_labels == expect_cohort_labels

        # check variants are consistent
        assert ds.sizes["variants"] == len(df_af)
        for v in expected_variant_vars:
            c = v.split("variant_")[1]
            actual = ds[v]
            expect = df_af[c]
            compare_series_like(actual, expect)

        # check frequencies are consistent
        for cohort_index, cohort_label in enumerate(ds["cohort_label"].values):
            actual_frq = ds["event_frequency"].values[:, cohort_index]
            expect_frq = df_af[f"frq_{cohort_label}"].values
            assert_allclose(actual_frq, expect_frq)


@pytest.mark.parametrize("region", ["2R", "X", ["3R", "X"], "3R:28,000,000-29,000,000"])
def test_gene_cnv_frequencies_advanced__region(region):
    _check_gene_cnv_frequencies_advanced(
        region=region,
    )


@pytest.mark.parametrize("area_by", ["country", "admin1_iso", "admin2_name"])
def test_gene_cnv_frequencies_advanced__area_by(area_by):
    _check_gene_cnv_frequencies_advanced(
        area_by=area_by,
    )


@pytest.mark.parametrize("period_by", ["year", "quarter", "month"])
def test_gene_cnv_frequencies_advanced__period_by(period_by):
    _check_gene_cnv_frequencies_advanced(
        period_by=period_by,
    )


@pytest.mark.parametrize(
    "sample_sets", ["AG1000G-BF-A", ["AG1000G-BF-A", "AG1000G-ML-A"], "3.0"]
)
def test_gene_cnv_frequencies_advanced__sample_sets(sample_sets):
    _check_gene_cnv_frequencies_advanced(
        sample_sets=sample_sets,
    )


@pytest.mark.parametrize(
    "sample_query",
    [
        "taxon in ['gambiae', 'coluzzii'] and country == 'Mali'",
        "taxon == 'arabiensis' and country in ['Uganda', 'Tanzania']",
    ],
)
def test_gene_cnv_frequencies_advanced__sample_query(sample_query):
    _check_gene_cnv_frequencies_advanced(
        sample_query=sample_query,
    )


@pytest.mark.parametrize("min_cohort_size", [10, 100])
def test_gene_cnv_frequencies_advanced__min_cohort_size(min_cohort_size):
    _check_gene_cnv_frequencies_advanced(
        min_cohort_size=min_cohort_size,
    )


@pytest.mark.parametrize(
    "variant_query",
    [
        None,
        "cnv_type == 'amp' and max_af > 0.05",
    ],
)
def test_gene_cnv_frequencies_advanced__variant_query(variant_query):
    _check_gene_cnv_frequencies_advanced(
        variant_query=variant_query,
    )


@pytest.mark.parametrize(
    "drop_invariant",
    [
        False,
        True,
    ],
)
def test_gene_cnv_frequencies_advanced__drop_invariant(drop_invariant):
    # noinspection PyTypeChecker
    _check_gene_cnv_frequencies_advanced(
        variant_query=None,
        drop_invariant=drop_invariant,
    )


@pytest.mark.parametrize(
    "max_coverage_variance",
    [None, 0.2],
)
def test_gene_cnv_frequencies_advanced__max_coverage_variance(max_coverage_variance):
    _check_gene_cnv_frequencies_advanced(
        max_coverage_variance=max_coverage_variance,
        sample_sets=["AG1000G-GM-A", "AG1000G-GM-B", "AG1000G-GM-C"],
    )


def test_gene_cnv_frequencies_advanced__multi_contig_x():
    # https://github.com/malariagen/malariagen-data-python/issues/166

    ag3 = setup_ag3(cohorts_analysis="20230516")

    ds1 = ag3.gene_cnv_frequencies_advanced(
        region="X",
        area_by="admin1_iso",
        period_by="year",
        sample_sets="AG1000G-BF-B",
        sample_query=None,
        min_cohort_size=10,
        variant_query=None,
        drop_invariant=False,
        max_coverage_variance=None,
    )

    ds2 = ag3.gene_cnv_frequencies_advanced(
        region=["2R", "X"],
        area_by="admin1_iso",
        period_by="year",
        sample_sets="AG1000G-BF-B",
        sample_query=None,
        min_cohort_size=10,
        variant_query=None,
        drop_invariant=False,
        max_coverage_variance=None,
    )
    loc_x = ds2["variant_contig"].values == "X"
    ds2 = ds2.isel(variants=loc_x)

    for v in ds1:
        a = ds1[v]
        b = ds2[v]
        compare_series_like(a, b)


def test_gene_cnv_frequencies_advanced__missing_samples():
    # https://github.com/malariagen/malariagen-data-python/issues/183

    ag3 = setup_ag3(cohorts_analysis="20230516", pre=True)

    ds = ag3.gene_cnv_frequencies_advanced(
        region="3L",
        sample_sets="1190-VO-GH-AMENGA-ETEGO-VMF00013",
        area_by="admin1_iso",
        period_by="year",
    )
    assert isinstance(ds, xr.Dataset)


def test_gene_cnv_frequencies_advanced__dup_samples():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    ds_dup = ag3.gene_cnv_frequencies_advanced(
        region="3L",
        area_by="admin1_iso",
        period_by="year",
        sample_sets=["AG1000G-BF-A", "AG1000G-BF-A"],
    )
    ds = ag3.gene_cnv_frequencies_advanced(
        region="3L",
        area_by="admin1_iso",
        period_by="year",
        sample_sets=["AG1000G-BF-A"],
    )
    assert ds.dims == ds_dup.dims


def test_h12_gwss():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    sample_query = "country == 'Ghana'"
    contig = "3L"
    analysis = "gamb_colu"
    sample_sets = "3.0"
    window_size = 1000

    x, h12 = ag3.h12_gwss(
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
    assert len(x) == 11354
    assert len(x) == len(h12)

    # check some values
    assert_allclose(x[0], 27701.195)
    assert_allclose(h12[11353], 0.17875)


def test_h1x_gwss():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    cohort1_query = "cohort_admin2_year == 'ML-2_Kati_colu_2014'"
    cohort2_query = "cohort_admin2_year == 'ML-2_Kati_gamb_2014'"
    sample_sets = "3.0"
    contig = "2L"
    analysis = "gamb_colu"
    window_size = 2000

    x, h1x = ag3.h1x_gwss(
        contig=contig,
        analysis=analysis,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        sample_sets=sample_sets,
        window_size=window_size,
        cohort_size=None,
    )

    # check data
    assert isinstance(x, np.ndarray)
    assert isinstance(h1x, np.ndarray)

    # check dimensions
    assert x.ndim == h1x.ndim == 1
    assert x.shape == h1x.shape

    # check some values
    assert_allclose(x[0], 36493.229, rtol=1e-5), x[0]
    assert_allclose(h1x[0], 0.067621, rtol=1e-5), h1x[0]
    assert np.all(h1x <= 1)
    assert np.all(h1x >= 0)


def test_ihs_gwss():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    sample_query = "country == 'Ghana'"
    contig = "3L"
    analysis = "gamb_colu"
    sample_sets = "3.0"
    window_size = 1000

    x, ihs = ag3.ihs_gwss(
        contig=contig,
        analysis=analysis,
        sample_query=sample_query,
        sample_sets=sample_sets,
        window_size=window_size,
        max_cohort_size=20,
    )

    assert isinstance(x, np.ndarray)
    assert isinstance(ihs, np.ndarray)

    # check dimensions
    assert len(x) == 395
    assert len(x) == len(ihs)

    # check some values
    assert_allclose(x[0], 510232.847)
    assert_allclose(ihs[:, 2][100], 2.3467595962486327)


def test_xpehh_gwss():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    cohort1_query = "country == 'Ghana'"
    cohort2_query = "country == 'Angola'"
    contig = "3L"
    analysis = "gamb_colu"
    sample_sets = "3.0"
    window_size = 1000

    x, xpehh = ag3.xpehh_gwss(
        contig=contig,
        analysis=analysis,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        sample_sets=sample_sets,
        window_size=window_size,
        max_cohort_size=20,
    )

    assert isinstance(x, np.ndarray)
    assert isinstance(xpehh, np.ndarray)

    # check dimensions
    assert len(x) == 399
    assert len(x) == len(xpehh)

    # check some values
    assert_allclose(x[0], 467448.348)
    assert_allclose(xpehh[:, 2][100], 0.4817561326426265)
