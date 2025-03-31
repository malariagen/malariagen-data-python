import numpy as np
import pandas as pd
import xarray as xr
import pytest
from numpy.testing import assert_allclose

from malariagen_data import Ag3, Region
from malariagen_data.util import locate_region, resolve_region

contigs = "2R", "2L", "3R", "3L", "X"


def setup_ag3(url="simplecache::gs://vo_agam_release_master_us_central1/", **kwargs):
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


@pytest.mark.parametrize(
    "inversion",
    ["2La", "2Rb", "2Rc_col", "X_x"],
)
def test_karyotyping(inversion):
    ag3 = setup_ag3(cohorts_analysis="20230516")

    if inversion == "X_x":
        with pytest.raises(ValueError):
            ag3.karyotype(
                inversion=inversion, sample_sets="AG1000G-GH", sample_query=None
            )
    else:
        df = ag3.karyotype(
            inversion=inversion, sample_sets="AG1000G-GH", sample_query=None
        )
        assert isinstance(df, pd.DataFrame)
        expected_cols = [
            "sample_id",
            "inversion",
            f"karyotype_{inversion}_mean",
            f"karyotype_{inversion}",
            "total_tag_snps",
        ]
        assert set(df.columns) == set(expected_cols)
        assert all(df[f"karyotype_{inversion}"].isin([0, 1, 2]))
        assert all(df[f"karyotype_{inversion}_mean"].between(0, 2))


@pytest.mark.parametrize(
    "inversions",
    ["2La", ["2Rb", "2Rc_col"], [], "X_x"],
)
def test_inversion_frequencies(inversions):
    ag3 = setup_ag3(cohorts_analysis="20230516")

    if inversions == "X_x" or inversions == []:
        with pytest.raises(ValueError):
            ag3.inversion_frequencies(
                inversions=inversions,
                cohorts="admin1_year",
                sample_sets="AG1000G-GH",
            )
    else:
        df = ag3.inversion_frequencies(
            inversions=inversions,
            cohorts="admin1_year",
            sample_sets="AG1000G-GH",
            min_cohort_size=10,
        )
        assert isinstance(df, pd.DataFrame)
        expected_cols = [
            "inversion",
            "allele",
            "label",
        ]
        df_samples = ag3.sample_metadata(sample_sets="AG1000G-GH")
        cohort_column = "cohort_admin1_year"
        cohort_counts = df_samples[cohort_column].value_counts()
        cohort_labels = cohort_counts[cohort_counts >= 10].index.to_list()
        frq_fields = ["frq_" + s for s in cohort_labels]
        expected_cols += frq_fields

        assert sorted(df.columns.tolist()) == sorted(expected_cols)
        for f in frq_fields:
            x = df[f]
            check_frequency(x)


@pytest.mark.parametrize(
    "inversions",
    ["2La", ["2Rb", "2Rc_col"], [], "X_x"],
)
def test_inversion_frequencies_include_counts(inversions):
    ag3 = setup_ag3(cohorts_analysis="20230516")

    if inversions == "X_x" or inversions == []:
        with pytest.raises(ValueError):
            ag3.inversion_frequencies(
                inversions=inversions,
                cohorts="admin1_year",
                sample_sets="AG1000G-GH",
                include_counts=True,
            )
    else:
        df = ag3.inversion_frequencies(
            inversions=inversions,
            cohorts="admin1_year",
            sample_sets="AG1000G-GH",
            min_cohort_size=10,
            include_counts=True,
        )
        assert isinstance(df, pd.DataFrame)
        expected_cols = [
            "inversion",
            "allele",
            "label",
        ]
        df_samples = ag3.sample_metadata(sample_sets="AG1000G-GH")
        cohort_column = "cohort_admin1_year"
        cohort_counts = df_samples[cohort_column].value_counts()
        cohort_labels = cohort_counts[cohort_counts >= 10].index.to_list()
        frq_fields = ["frq_" + s for s in cohort_labels]
        expected_cols += frq_fields
        count_fields = ["count_" + s for s in cohort_labels]
        expected_cols += count_fields
        nobs_fields = ["nobs_" + s for s in cohort_labels]
        expected_cols += nobs_fields

        assert sorted(df.columns.tolist()) == sorted(expected_cols)
        for f in frq_fields:
            x = df[f]
            check_frequency(x)


@pytest.mark.parametrize(
    "cohorts", ["admin1_year", "admin2_month", "country", "foobar"]
)
def test_inversion_frequencies_cohorts(cohorts):
    ag3 = setup_ag3(cohorts_analysis="20230516")

    if cohorts == "foobar":
        with pytest.raises(ValueError):
            ag3.inversion_frequencies(
                inversions="2Ru",
                cohorts=cohorts,
                sample_sets="AG1000G-GH",
            )
    else:
        df = ag3.inversion_frequencies(
            inversions="2Ru",
            cohorts=cohorts,
            sample_sets="AG1000G-GH",
            min_cohort_size=10,
        )
        assert isinstance(df, pd.DataFrame)
        expected_cols = [
            "inversion",
            "allele",
            "label",
        ]
        df_samples = ag3.sample_metadata(sample_sets="AG1000G-GH")
        if cohorts in df_samples.columns:
            cohort_column = cohorts
        else:
            cohort_column = "cohort_" + cohorts
        cohort_counts = df_samples[cohort_column].value_counts()
        cohort_labels = cohort_counts[cohort_counts >= 10].index.to_list()
        frq_fields = ["frq_" + s for s in cohort_labels]
        expected_cols += frq_fields

        assert sorted(df.columns.tolist()) == sorted(expected_cols)
        for f in frq_fields:
            x = df[f]
            check_frequency(x)


@pytest.mark.parametrize("min_cohort_size", [0, 10, 100])
def test_inversion_frequencies_min_cohort_size(min_cohort_size):
    ag3 = setup_ag3(cohorts_analysis="20230516")

    # Figure out expected cohort labels.
    df_samples = ag3.sample_metadata(sample_sets="AG1000G-GH")
    cohort_counts = df_samples["cohort_admin1_year"].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    if len(cohort_labels) == 0:
        # No cohorts, expect error.
        with pytest.raises(ValueError):
            ag3.inversion_frequencies(
                inversions="2Ru",
                cohorts="admin1_year",
                sample_sets="AG1000G-GH",
                min_cohort_size=min_cohort_size,
            )
            return
    else:
        df = ag3.inversion_frequencies(
            inversions="2Ru",
            cohorts="admin1_year",
            sample_sets="AG1000G-GH",
            min_cohort_size=min_cohort_size,
        )
        assert isinstance(df, pd.DataFrame)
        expected_cols = [
            "inversion",
            "allele",
            "label",
        ]
        frq_fields = ["frq_" + s for s in cohort_labels]
        expected_cols += frq_fields

        assert sorted(df.columns.tolist()) == sorted(expected_cols)
        for f in frq_fields:
            x = df[f]
            check_frequency(x)


@pytest.mark.parametrize("admin1_name", ["Central Region", "Eastern Region"])
def test_inversion_frequencies_sample_query(admin1_name):
    ag3 = setup_ag3(cohorts_analysis="20230516")

    df = ag3.inversion_frequencies(
        inversions="2Ru",
        cohorts="admin1_year",
        sample_query=f"admin1_name == '{admin1_name}'",
        sample_sets="AG1000G-GH",
        min_cohort_size=10,
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = [
        "inversion",
        "allele",
        "label",
    ]
    df_samples = ag3.sample_metadata(
        sample_sets="AG1000G-GH", sample_query=f"admin1_name == '{admin1_name}'"
    )
    cohort_column = "cohort_admin1_year"
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= 10].index.to_list()
    frq_fields = ["frq_" + s for s in cohort_labels]
    expected_cols += frq_fields

    assert sorted(df.columns.tolist()) == sorted(expected_cols)
    for f in frq_fields:
        x = df[f]
        check_frequency(x)

    df = ag3.inversion_frequencies(
        inversions="2Ru",
        cohorts="admin1_year",
        sample_query=f"admin1_name != '{admin1_name}'",
        sample_sets="AG1000G-GH",
        min_cohort_size=10,
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = [
        "inversion",
        "allele",
        "label",
    ]
    df_samples = ag3.sample_metadata(
        sample_sets="AG1000G-GH", sample_query=f"admin1_name != '{admin1_name}'"
    )
    cohort_column = "cohort_admin1_year"
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= 10].index.to_list()
    frq_fields = ["frq_" + s for s in cohort_labels]
    expected_cols += frq_fields

    assert sorted(df.columns.tolist()) == sorted(expected_cols)
    for f in frq_fields:
        x = df[f]
        check_frequency(x)


@pytest.mark.parametrize("admin1_name", ["Central Region", "Eastern Region"])
def test_inversion_frequencies_sample_query_options(admin1_name):
    ag3 = setup_ag3(cohorts_analysis="20230516")

    df = ag3.inversion_frequencies(
        inversions="2Ru",
        cohorts="admin1_year",
        sample_query="admin1_name == @admin1_name",
        sample_query_options={
            "local_dict": {
                "admin1_name": admin1_name,
            }
        },
        sample_sets="AG1000G-GH",
        min_cohort_size=10,
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = [
        "inversion",
        "allele",
        "label",
    ]
    df_samples = ag3.sample_metadata(
        sample_sets="AG1000G-GH",
        sample_query="admin1_name == @admin1_name",
        sample_query_options={
            "local_dict": {
                "admin1_name": admin1_name,
            }
        },
    )
    cohort_column = "cohort_admin1_year"
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= 10].index.to_list()
    frq_fields = ["frq_" + s for s in cohort_labels]
    expected_cols += frq_fields

    assert sorted(df.columns.tolist()) == sorted(expected_cols)
    for f in frq_fields:
        x = df[f]
        check_frequency(x)

    df = ag3.inversion_frequencies(
        inversions="2Ru",
        cohorts="admin1_year",
        sample_query="admin1_name != @admin1_name",
        sample_query_options={
            "local_dict": {
                "admin1_name": admin1_name,
            }
        },
        sample_sets="AG1000G-GH",
        min_cohort_size=10,
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = [
        "inversion",
        "allele",
        "label",
    ]
    df_samples = ag3.sample_metadata(
        sample_sets="AG1000G-GH",
        sample_query="admin1_name != @admin1_name",
        sample_query_options={
            "local_dict": {
                "admin1_name": admin1_name,
            }
        },
    )
    cohort_column = "cohort_admin1_year"
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= 10].index.to_list()
    frq_fields = ["frq_" + s for s in cohort_labels]
    expected_cols += frq_fields

    assert sorted(df.columns.tolist()) == sorted(expected_cols)
    for f in frq_fields:
        x = df[f]
        check_frequency(x)


@pytest.mark.parametrize(
    "cohorts",
    [
        {
            "in": "admin1_name in ['Central Region', 'Eastern Region']",
            "out": "admin1_name not in ['Central Region', 'Eastern Region']",
        },
        {"gam": "taxon == 'gambiae'", "col": "taxon == 'coluzzii'"},
    ],
)
def test_inversion_frequencies_dict_cohorts(cohorts):
    ag3 = setup_ag3(cohorts_analysis="20230516")

    df = ag3.inversion_frequencies(
        inversions="2Ru",
        cohorts=cohorts,
        sample_sets="AG1000G-GH",
        min_cohort_size=10,
    )
    assert isinstance(df, pd.DataFrame)
    expected_cols = [
        "inversion",
        "allele",
        "label",
    ]

    frq_fields = ["frq_" + s for s in cohorts.keys()]
    expected_cols += frq_fields

    assert sorted(df.columns.tolist()) == sorted(expected_cols)
    for f in frq_fields:
        x = df[f]
        check_frequency(x)


@pytest.mark.parametrize("area_by", ["country", "admin1_iso", "admin2_name"])
def test_inversion_frequencies_advanced_with_area_by(
    area_by,
):
    check_inversion_frequencies_advanced(
        area_by=area_by,
    )


@pytest.mark.parametrize("period_by", ["year", "quarter", "month"])
def test_inversion_frequencies_advanced_with_period_by(
    period_by,
):
    check_inversion_frequencies_advanced(
        period_by=period_by,
    )


@pytest.mark.parametrize("taxon", ["coluzzii", "gambiae"])
def test_inversion_frequencies_advanced_with_sample_query(
    taxon,
):
    check_inversion_frequencies_advanced(
        sample_query=f"taxon == '{taxon}'",
        min_cohort_size=0,
    )


@pytest.mark.parametrize("taxon", ["coluzzii", "gambiae"])
def test_inversion_frequencies_advanced_with_sample_query_options(
    taxon,
):
    check_inversion_frequencies_advanced(
        sample_query="taxon == @taxon_name",
        sample_query_options={
            "local_dict": {
                "taxon_name": taxon,
            }
        },
        min_cohort_size=0,
    )


@pytest.mark.parametrize("min_cohort_size", [0, 10, 100])
def test_inversion_frequencies_advanced_with_min_cohort_size(
    min_cohort_size,
):
    if min_cohort_size <= 10:
        check_inversion_frequencies_advanced(
            min_cohort_size=min_cohort_size,
        )
    else:
        with pytest.raises(ValueError):
            check_inversion_frequencies_advanced(
                min_cohort_size=min_cohort_size,
            )


def check_frequency(x):
    loc_nan = np.isnan(x)
    assert np.all(x[~loc_nan] >= 0)
    assert np.all(x[~loc_nan] <= 1)


def check_inversion_frequencies_advanced(
    *,
    inversions="2Rb",
    area_by="admin1_iso",
    period_by="year",
    sample_sets="AG1000G-GH",
    sample_query=None,
    sample_query_options=None,
    min_cohort_size=10,
):
    ag3 = setup_ag3(cohorts_analysis="20230516")

    # Run function under test.
    ds = ag3.inversion_frequencies_advanced(
        inversions=inversions,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        min_cohort_size=min_cohort_size,
    )

    # Check the result.
    assert isinstance(ds, xr.Dataset)
    assert set(ds.dims) == {"cohorts", "variants"}

    # Check variant variables.
    expected_variant_vars = [
        "variant_label",
    ]
    for v in expected_variant_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants",)

    # Check cohort variables.
    expected_cohort_vars = [
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
    ]
    for v in expected_cohort_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("cohorts",)

    # Check event variables.
    expected_event_vars = [
        "event_count",
        "event_nobs",
        "event_frequency",
        "event_frequency_ci_low",
        "event_frequency_ci_upp",
    ]
    for v in expected_event_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants", "cohorts")

    # Sanity check for frequency values.
    x = ds["event_frequency"].values
    check_frequency(x)

    # Sanity check area values.
    df_samples = ag3.sample_metadata(
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
    )
    expected_area_values = np.unique(df_samples[area_by].dropna().values)
    area_values = ds["cohort_area"].values
    # N.B., some areas may not end up in final dataset if cohort
    # size is too small, so do a set membership test
    for a in area_values:
        assert a in expected_area_values

    # Sanity checks for period values.
    period_values = ds["cohort_period"].values
    if period_by == "year":
        expected_freqstr = "Y-DEC"
    elif period_by == "month":
        expected_freqstr = "M"
    elif period_by == "quarter":
        expected_freqstr = "Q-DEC"
    else:
        assert False, "not implemented"
    for p in period_values:
        assert isinstance(p, pd.Period)
        assert p.freqstr == expected_freqstr

    # Sanity check cohort sizes.
    cohort_size_values = ds["cohort_size"].values
    for s in cohort_size_values:
        assert s >= min_cohort_size

    if area_by == "admin1_iso" and period_by == "year":
        # Here we test the behaviour of the function when grouping by admin level
        # 1 and year. We can do some more in-depth testing in this case because
        # we can compare results directly against the simpler snp_allele_frequencies()
        # function with the admin1_year cohorts.

        # Check consistency with the basic snp allele frequencies method.
        df_if = ag3.inversion_frequencies(
            inversions=inversions,
            cohorts="admin1_year",
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            min_cohort_size=min_cohort_size,
        )

        # Check cohorts are consistent.
        expect_cohort_labels = sorted(
            [c.split("frq_")[1] for c in df_if.columns if c.startswith("frq_")]
        )
        cohort_labels = sorted(ds["cohort_label"].values)
        assert cohort_labels == expect_cohort_labels

        # Check variants are consistent.
        assert ds.sizes["variants"] == len(df_if)
