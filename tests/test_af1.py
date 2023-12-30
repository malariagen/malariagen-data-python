import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal

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


# noinspection PyDefaultArgument
def _check_snp_allele_frequencies_advanced(
    transcript="LOC125767311_t2",
    area_by="admin1_iso",
    period_by="year",
    sample_sets=[
        "1229-VO-GH-DADZIE-VMF00095",
        "1240-VO-CD-KOEKEMOER-VMF00099",
        "1240-VO-MZ-KOEKEMOER-VMF00101",
    ],
    sample_query=None,
    min_cohort_size=10,
    nobs_mode="called",
    variant_query="max_af > 0.02",
):
    af1 = setup_af1(cohorts_analysis="20221129")

    ds = af1.snp_allele_frequencies_advanced(
        transcript=transcript,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        sample_query=sample_query,
        min_cohort_size=min_cohort_size,
        nobs_mode=nobs_mode,
        variant_query=variant_query,
    )

    assert isinstance(ds, xr.Dataset)

    # noinspection PyTypeChecker
    assert sorted(ds.dims) == ["cohorts", "variants"]

    expected_variant_vars = (
        "variant_label",
        "variant_contig",
        "variant_position",
        "variant_ref_allele",
        "variant_alt_allele",
        "variant_max_af",
        "variant_pass_funestus",
        "variant_transcript",
        "variant_effect",
        "variant_impact",
        "variant_ref_codon",
        "variant_alt_codon",
        "variant_ref_aa",
        "variant_alt_aa",
        "variant_aa_pos",
        "variant_aa_change",
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
    df_samples = af1.sample_metadata(sample_sets=sample_sets)
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

    if area_by == "admin1_iso" and period_by == "year" and nobs_mode == "called":
        # Here we test the behaviour of the function when grouping by admin level
        # 1 and year. We can do some more in-depth testing in this case because
        # we can compare results directly against the simpler snp_allele_frequencies()
        # function with the admin1_year cohorts.

        # check consistency with the basic snp allele frequencies method
        df_af = af1.snp_allele_frequencies(
            transcript=transcript,
            cohorts="admin1_year",
            sample_sets=sample_sets,
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
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
            _compare_series_like(actual, expect)

        # check frequencies are consistent
        for cohort_index, cohort_label in enumerate(ds["cohort_label"].values):
            actual_frq = ds["event_frequency"].values[:, cohort_index]
            expect_frq = df_af[f"frq_{cohort_label}"].values
            assert_allclose(actual_frq, expect_frq)


# noinspection PyDefaultArgument
def _check_aa_allele_frequencies_advanced(
    transcript="LOC125767311_t2",
    area_by="admin1_iso",
    period_by="year",
    sample_sets=[
        "1229-VO-GH-DADZIE-VMF00095",
        "1240-VO-CD-KOEKEMOER-VMF00099",
        "1240-VO-MZ-KOEKEMOER-VMF00101",
    ],
    sample_query=None,
    min_cohort_size=10,
    nobs_mode="called",
    variant_query="max_af > 0.02",
):
    af1 = setup_af1(cohorts_analysis="20221129")

    ds = af1.aa_allele_frequencies_advanced(
        transcript=transcript,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        sample_query=sample_query,
        min_cohort_size=min_cohort_size,
        nobs_mode=nobs_mode,
        variant_query=variant_query,
    )

    assert isinstance(ds, xr.Dataset)

    # noinspection PyTypeChecker
    assert sorted(ds.dims) == ["cohorts", "variants"]

    expected_variant_vars = (
        "variant_label",
        "variant_contig",
        "variant_position",
        "variant_max_af",
        "variant_transcript",
        "variant_effect",
        "variant_impact",
        "variant_ref_aa",
        "variant_alt_aa",
        "variant_aa_pos",
        "variant_aa_change",
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
    df_samples = af1.sample_metadata(sample_sets=sample_sets)
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

    if area_by == "admin1_iso" and period_by == "year" and nobs_mode == "called":
        # Here we test the behaviour of the function when grouping by admin level
        # 1 and year. We can do some more in-depth testing in this case because
        # we can compare results directly against the simpler aa_allele_frequencies()
        # function with the admin1_year cohorts.

        # check consistency with the basic snp allele frequencies method
        df_af = af1.aa_allele_frequencies(
            transcript=transcript,
            cohorts="admin1_year",
            sample_sets=sample_sets,
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
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
            _compare_series_like(actual, expect)

        # check frequencies are consistent
        for cohort_index, cohort_label in enumerate(ds["cohort_label"].values):
            print(cohort_label)
            actual_frq = ds["event_frequency"].values[:, cohort_index]
            expect_frq = df_af[f"frq_{cohort_label}"].values
            assert_allclose(actual_frq, expect_frq)


# Here we don't explore the full matrix, but vary one parameter at a time, otherwise
# the test suite would take too long to run.


@pytest.mark.parametrize(
    "transcript", ["LOC125767311_t2", "LOC125761549_t5", "LOC125761549_t7"]
)
def test_allele_frequencies_advanced__transcript(transcript):
    _check_snp_allele_frequencies_advanced(
        transcript=transcript,
    )
    _check_aa_allele_frequencies_advanced(
        transcript=transcript,
    )


@pytest.mark.parametrize("area_by", ["country", "admin1_iso", "admin2_name"])
def test_allele_frequencies_advanced__area_by(area_by):
    _check_snp_allele_frequencies_advanced(
        area_by=area_by,
    )
    _check_aa_allele_frequencies_advanced(
        area_by=area_by,
    )


@pytest.mark.parametrize("period_by", ["year", "quarter", "month"])
def test_allele_frequencies_advanced__period_by(period_by):
    _check_snp_allele_frequencies_advanced(
        period_by=period_by,
    )
    _check_aa_allele_frequencies_advanced(
        period_by=period_by,
    )


@pytest.mark.parametrize(
    "sample_sets",
    [
        "1229-VO-GH-DADZIE-VMF00095",
        ["1240-VO-CD-KOEKEMOER-VMF00099", "1240-VO-MZ-KOEKEMOER-VMF00101"],
        "1.0",
    ],
)
def test_allele_frequencies_advanced__sample_sets(sample_sets):
    _check_snp_allele_frequencies_advanced(
        sample_sets=sample_sets,
    )
    _check_aa_allele_frequencies_advanced(
        sample_sets=sample_sets,
    )


def test_allele_frequencies_advanced__sample_query():
    _check_snp_allele_frequencies_advanced(
        sample_query="taxon == 'funestus' and country in ['Ghana', 'Gabon']",
    )
    # noinspection PyTypeChecker
    _check_aa_allele_frequencies_advanced(
        sample_query="taxon == 'funestus' and country in ['Ghana', 'Gabon']",
        variant_query=None,
    )


@pytest.mark.parametrize("min_cohort_size", [10, 40])
def test_allele_frequencies_advanced__min_cohort_size(min_cohort_size):
    _check_snp_allele_frequencies_advanced(
        min_cohort_size=min_cohort_size,
    )
    _check_aa_allele_frequencies_advanced(
        min_cohort_size=min_cohort_size,
    )


@pytest.mark.parametrize(
    "variant_query",
    [
        None,
        "effect == 'NON_SYNONYMOUS_CODING' and max_af > 0.05",
        "effect == 'foobar'",  # no variants
    ],
)
def test_allele_frequencies_advanced__variant_query(variant_query):
    _check_snp_allele_frequencies_advanced(
        variant_query=variant_query,
    )
    _check_aa_allele_frequencies_advanced(
        variant_query=variant_query,
    )


@pytest.mark.parametrize("nobs_mode", ["called", "fixed"])
def test_allele_frequencies_advanced__nobs_mode(nobs_mode):
    _check_snp_allele_frequencies_advanced(
        nobs_mode=nobs_mode,
    )
    _check_aa_allele_frequencies_advanced(
        nobs_mode=nobs_mode,
    )


# TODO: this function is a verbatim duplicate, from test_ag3.py
def _compare_series_like(actual, expect):
    # compare pandas series-like objects for equality or floating point
    # similarity, handling missing values appropriately

    # handle object arrays, these don't get nans compared properly
    t = actual.dtype
    if t == object:
        expect = expect.fillna("NA")
        actual = actual.fillna("NA")

    if t.kind == "f":
        assert_allclose(actual.values, expect.values)
    else:
        assert_array_equal(actual.values, expect.values)


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


def test_g123_gwss():
    af1 = setup_af1(cohorts_analysis="20230823")
    sample_query = "country == 'Ghana'"
    contig = "3RL"
    site_mask = "funestus"
    sample_sets = "1.0"
    window_size = 10_000

    x, g123 = af1.g123_gwss(
        contig=contig,
        site_mask=site_mask,
        sites=site_mask,
        sample_query=sample_query,
        sample_sets=sample_sets,
        window_size=window_size,
        min_cohort_size=20,
        max_cohort_size=30,
    )

    # check dataset
    assert isinstance(x, np.ndarray)
    assert isinstance(g123, np.ndarray)

    # check dimensions
    assert len(x) == 1584
    assert len(x) == len(g123)

    # check some values
    assert_allclose(x[0], 253398.2095)
    assert_allclose(g123[0], 0.04)
