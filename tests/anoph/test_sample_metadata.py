import ipyleaflet
import numpy as np
import pandas as pd
import pytest
from pytest_cases import parametrize_with_cases
from typeguard import TypeCheckError

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.sample_metadata import AnophelesSampleMetadata


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesSampleMetadata(
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
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesSampleMetadata(
        url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        gcs_url=_af1.GCS_URL,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
    )


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


def general_metadata_expected_columns():
    return {
        "sample_id": "O",
        "partner_sample_id": "O",
        "contributor": "O",
        "country": "O",
        "location": "O",
        "year": "i",
        "month": "i",
        "latitude": "f",
        "longitude": "f",
        "sex_call": "O",
        "sample_set": "O",
        "release": "O",
        "quarter": "i",
    }


def validate_metadata(df, expected_columns):
    # Check column names.
    expected_column_names = list(expected_columns.keys())
    assert df.columns.to_list() == expected_column_names

    # Check column types.
    for c in df.columns:
        assert df[c].dtype.kind == expected_columns[c]


@pytest.mark.parametrize(
    "sample_set", ["AG1000G-AO", "AG1000G-BF-A", "1177-VO-ML-LEHMANN-VMF00004"]
)
def test_general_metadata__ag3_single_sample_set(ag3_sim_api, sample_set):
    df = ag3_sim_api.general_metadata(sample_sets=sample_set)
    validate_metadata(df, general_metadata_expected_columns())

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


@pytest.mark.parametrize(
    "sample_set",
    [
        "1229-VO-GH-DADZIE-VMF00095",
        "1230-VO-GA-CF-AYALA-VMF00045",
        "1231-VO-MULTI-WONDJI-VMF00043",
    ],
)
def test_general_metadata__af1_single_sample_set(af1_sim_api, sample_set):
    df = af1_sim_api.general_metadata(sample_sets=sample_set)
    validate_metadata(df, general_metadata_expected_columns())

    # Check number of rows.
    sample_count = af1_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


def test_general_metadata__ag3_multiple_sample_sets(ag3_sim_api):
    sample_sets = ["AG1000G-AO", "1177-VO-ML-LEHMANN-VMF00004"]
    df = ag3_sim_api.general_metadata(sample_sets=sample_sets)
    validate_metadata(df, general_metadata_expected_columns())

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_general_metadata__af1_multiple_sample_sets(af1_sim_api):
    sample_sets = ["1230-VO-GA-CF-AYALA-VMF00045", "1231-VO-MULTI-WONDJI-VMF00043"]
    df = af1_sim_api.general_metadata(sample_sets=sample_sets)
    validate_metadata(df, general_metadata_expected_columns())

    # Check number of rows.
    sample_count = af1_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_general_metadata__ag3_release(ag3_sim_api):
    release = "3.0"
    df = ag3_sim_api.general_metadata(sample_sets=release)
    validate_metadata(df, general_metadata_expected_columns())

    # Check number of rows.
    expected_len = ag3_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def test_general_metadata__af1_release(af1_sim_api):
    release = "1.0"
    df = af1_sim_api.general_metadata(sample_sets=release)
    validate_metadata(df, general_metadata_expected_columns())

    # Check number of rows.
    expected_len = af1_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def aim_metadata_expected_columns():
    return {
        "sample_id": "O",
        "aim_species_fraction_arab": "f",
        "aim_species_fraction_colu": "f",
        "aim_species_fraction_colu_no2l": "f",
        "aim_species_gambcolu_arabiensis": "O",
        "aim_species_gambiae_coluzzii": "O",
        "aim_species": "O",
    }


def validate_aim_metadata(df):
    validate_metadata(df, aim_metadata_expected_columns())

    # Check some values.
    expected_species = {
        "gambiae",
        "coluzzii",
        "arabiensis",
        "intermediate_gambcolu_arabiensis",
        "intermediate_gambiae_coluzzii",
    }
    for v in df["aim_species"]:
        if isinstance(v, str):
            assert v in expected_species
        else:
            assert np.isnan(v)


@pytest.mark.parametrize(
    "sample_set", ["AG1000G-AO", "AG1000G-BF-A", "1177-VO-ML-LEHMANN-VMF00004"]
)
def test_aim_metadata__ag3_single_sample_set(ag3_sim_api, sample_set):
    df = ag3_sim_api.aim_metadata(sample_sets=sample_set)
    validate_aim_metadata(df)

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


def test_aim_metadata__ag3_multiple_sample_sets(ag3_sim_api):
    sample_sets = ["AG1000G-AO", "1177-VO-ML-LEHMANN-VMF00004"]
    df = ag3_sim_api.aim_metadata(sample_sets=sample_sets)
    validate_aim_metadata(df)

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_aim_metadata__ag3_release(ag3_sim_api):
    release = "3.0"
    df = ag3_sim_api.aim_metadata(sample_sets=release)
    validate_aim_metadata(df)

    # Check number of rows.
    expected_len = ag3_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def cohorts_metadata_expected_columns(has_quarter):
    if has_quarter:
        return {
            "sample_id": "O",
            "country_iso": "O",
            "admin1_name": "O",
            "admin1_iso": "O",
            "admin2_name": "O",
            "taxon": "O",
            "cohort_admin1_year": "O",
            "cohort_admin1_month": "O",
            "cohort_admin1_quarter": "O",
            "cohort_admin2_year": "O",
            "cohort_admin2_month": "O",
            "cohort_admin2_quarter": "O",
        }
    else:
        return {
            "sample_id": "O",
            "country_iso": "O",
            "admin1_name": "O",
            "admin1_iso": "O",
            "admin2_name": "O",
            "taxon": "O",
            "cohort_admin1_year": "O",
            "cohort_admin1_month": "O",
            "cohort_admin2_year": "O",
            "cohort_admin2_month": "O",
        }


def validate_cohorts_metadata(df, has_quarter):
    # N.B., older cohorts metadata only has cohorts by year and month.
    # Newer cohorts metadata also has cohorts by quarter.
    expected_columns = cohorts_metadata_expected_columns(has_quarter=has_quarter)
    validate_metadata(df, expected_columns)


@pytest.mark.parametrize(
    "sample_set", ["AG1000G-AO", "AG1000G-BF-A", "1177-VO-ML-LEHMANN-VMF00004"]
)
def test_cohorts_metadata__ag3_single_sample_set(ag3_sim_api, sample_set):
    df = ag3_sim_api.cohorts_metadata(sample_sets=sample_set)
    validate_cohorts_metadata(df, has_quarter=True)

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


@pytest.mark.parametrize(
    "sample_set",
    [
        "1229-VO-GH-DADZIE-VMF00095",
        "1230-VO-GA-CF-AYALA-VMF00045",
        "1231-VO-MULTI-WONDJI-VMF00043",
    ],
)
def test_cohorts_metadata__af1_single_sample_set(af1_sim_api, sample_set):
    df = af1_sim_api.cohorts_metadata(sample_sets=sample_set)
    validate_cohorts_metadata(df, has_quarter=False)

    # Check number of rows.
    sample_count = af1_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


def test_cohorts_metadata__ag3_multiple_sample_sets(ag3_sim_api):
    sample_sets = ["AG1000G-AO", "1177-VO-ML-LEHMANN-VMF00004"]
    df = ag3_sim_api.cohorts_metadata(sample_sets=sample_sets)
    validate_cohorts_metadata(df, has_quarter=True)

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_cohorts_metadata__af1_multiple_sample_sets(af1_sim_api):
    sample_sets = ["1230-VO-GA-CF-AYALA-VMF00045", "1231-VO-MULTI-WONDJI-VMF00043"]
    df = af1_sim_api.cohorts_metadata(sample_sets=sample_sets)
    validate_cohorts_metadata(df, has_quarter=False)

    # Check number of rows.
    sample_count = af1_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_cohorts_metadata__ag3_release(ag3_sim_api):
    release = "3.0"
    df = ag3_sim_api.cohorts_metadata(sample_sets=release)
    validate_cohorts_metadata(df, has_quarter=True)

    # Check number of rows.
    expected_len = ag3_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def test_cohorts_metadata__af1_release(af1_sim_api):
    release = "1.0"
    df = af1_sim_api.cohorts_metadata(sample_sets=release)
    validate_cohorts_metadata(df, has_quarter=False)

    # Check number of rows.
    expected_len = af1_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def sample_metadata_expected_columns(has_aims, has_cohorts_by_quarter):
    expected_columns = general_metadata_expected_columns()
    if has_aims:
        expected_columns.update(aim_metadata_expected_columns())
    expected_columns.update(
        cohorts_metadata_expected_columns(has_quarter=has_cohorts_by_quarter)
    )
    return expected_columns


@pytest.mark.parametrize(
    "sample_set", ["AG1000G-AO", "AG1000G-BF-A", "1177-VO-ML-LEHMANN-VMF00004"]
)
def test_sample_metadata__ag3_single_sample_set(ag3_sim_api, sample_set):
    df = ag3_sim_api.sample_metadata(sample_sets=sample_set)
    validate_metadata(
        df, sample_metadata_expected_columns(has_aims=True, has_cohorts_by_quarter=True)
    )

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


@pytest.mark.parametrize(
    "sample_set",
    [
        "1229-VO-GH-DADZIE-VMF00095",
        "1230-VO-GA-CF-AYALA-VMF00045",
        "1231-VO-MULTI-WONDJI-VMF00043",
    ],
)
def test_sample_metadata__af1_single_sample_set(af1_sim_api, sample_set):
    df = af1_sim_api.sample_metadata(sample_sets=sample_set)
    validate_metadata(
        df,
        sample_metadata_expected_columns(has_aims=False, has_cohorts_by_quarter=False),
    )

    # Check number of rows.
    sample_count = af1_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


def test_sample_metadata__ag3_multiple_sample_sets(ag3_sim_api):
    sample_sets = ["AG1000G-AO", "1177-VO-ML-LEHMANN-VMF00004"]
    df = ag3_sim_api.sample_metadata(sample_sets=sample_sets)
    validate_metadata(
        df, sample_metadata_expected_columns(has_aims=True, has_cohorts_by_quarter=True)
    )

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_sample_metadata__af1_multiple_sample_sets(af1_sim_api):
    sample_sets = ["1230-VO-GA-CF-AYALA-VMF00045", "1231-VO-MULTI-WONDJI-VMF00043"]
    df = af1_sim_api.sample_metadata(sample_sets=sample_sets)
    validate_metadata(
        df,
        sample_metadata_expected_columns(has_aims=False, has_cohorts_by_quarter=False),
    )

    # Check number of rows.
    sample_count = af1_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_sample_metadata__ag3_release(ag3_sim_api):
    release = "3.0"
    df = ag3_sim_api.sample_metadata(sample_sets=release)
    validate_metadata(
        df, sample_metadata_expected_columns(has_aims=True, has_cohorts_by_quarter=True)
    )

    # Check number of rows.
    expected_len = ag3_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def test_sample_metadata__af1_release(af1_sim_api):
    release = "1.0"
    df = af1_sim_api.sample_metadata(sample_sets=release)
    validate_metadata(
        df,
        sample_metadata_expected_columns(has_aims=False, has_cohorts_by_quarter=False),
    )

    # Check number of rows.
    expected_len = af1_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def test_sample_metadata__ag3_query(ag3_sim_api):
    df = ag3_sim_api.sample_metadata(sample_query="country == 'Burkina Faso'")
    validate_metadata(
        df, sample_metadata_expected_columns(has_aims=True, has_cohorts_by_quarter=True)
    )
    assert (df["country"] == "Burkina Faso").all()


@parametrize_with_cases("fixture,api", cases=".")
def test_extra_metadata_errors(fixture, api):
    # Bad type.
    with pytest.raises(TypeCheckError):
        api.add_extra_metadata(data="foo")

    bad_data = pd.DataFrame({"foo": [1, 2, 3], "bar": ["a", "b", "c"]})

    # Missing sample identifier column.
    with pytest.raises(ValueError):
        api.add_extra_metadata(data=bad_data)

    # Invalid sample identifier column.
    with pytest.raises(ValueError):
        api.add_extra_metadata(data=bad_data, on="foo")

    # Duplicate identifiers.
    df_samples = api.sample_metadata()
    sample_id = df_samples["sample_id"].values
    data_with_dups = pd.DataFrame(
        {
            "sample_id": [sample_id[0], sample_id[0], sample_id[1]],
            "foo": [1, 2, 3],
            "bar": ["a", "b", "c"],
        }
    )
    with pytest.raises(ValueError):
        api.add_extra_metadata(data=data_with_dups)

    # No matching samples.
    data_no_matches = pd.DataFrame(
        {"sample_id": ["x", "y", "z"], "foo": [1, 2, 3], "bar": ["a", "b", "c"]}
    )
    with pytest.raises(ValueError):
        api.add_extra_metadata(data=data_no_matches)


@pytest.mark.parametrize(
    "on",
    [
        "sample_id",
        "partner_sample_id",
    ],
)
@parametrize_with_cases("fixture,api", cases=".")
def test_extra_metadata(fixture, api, on):
    # Load vanilla metadata.
    df_samples = api.sample_metadata()
    sample_id = df_samples[on].values

    # Partially overlapping data.
    extra1 = pd.DataFrame(
        {
            on: [sample_id[0], sample_id[1], "spam"],
            "foo": [1, 2, 3],
            "bar": ["a", "b", "c"],
        }
    )
    extra2 = pd.DataFrame(
        {
            on: [sample_id[2], sample_id[3], "eggs"],
            "baz": [True, False, True],
            "qux": [42, 84, 126],
        }
    )
    api.add_extra_metadata(data=extra1, on=on)
    api.add_extra_metadata(data=extra2, on=on)
    df_samples_extra = api.sample_metadata()
    assert "foo" in df_samples_extra.columns
    assert "bar" in df_samples_extra.columns
    assert "baz" in df_samples_extra.columns
    assert "qux" in df_samples_extra.columns
    assert df_samples_extra.columns.tolist() == (
        df_samples.columns.tolist() + ["foo", "bar", "baz", "qux"]
    )
    assert len(df_samples_extra) == len(df_samples)
    df_samples_extra = df_samples_extra.set_index(on)
    rec = df_samples_extra.loc[sample_id[0]]
    assert rec["foo"] == 1
    assert rec["bar"] == "a"
    assert np.isnan(rec["baz"])
    assert np.isnan(rec["qux"])
    rec = df_samples_extra.loc[sample_id[1]]
    assert rec["foo"] == 2
    assert rec["bar"] == "b"
    assert np.isnan(rec["baz"])
    assert np.isnan(rec["qux"])
    rec = df_samples_extra.loc[sample_id[2]]
    assert np.isnan(rec["foo"])
    assert np.isnan(rec["bar"])
    assert rec["baz"]
    assert rec["qux"] == 42
    rec = df_samples_extra.loc[sample_id[3]]
    assert np.isnan(rec["foo"])
    assert np.isnan(rec["bar"])
    assert not rec["baz"]
    assert rec["qux"] == 84
    with pytest.raises(KeyError):
        _ = df_samples_extra.loc["spam"]
    with pytest.raises(KeyError):
        _ = df_samples_extra.loc["eggs"]

    # Clear extra metadata.
    api.clear_extra_metadata()
    df_samples_cleared = api.sample_metadata()
    assert df_samples_cleared.columns.tolist() == df_samples.columns.tolist()
    assert len(df_samples_cleared) == len(df_samples)


@parametrize_with_cases("fixture,api", cases=".")
def test_count_samples(fixture, api):
    df = api.count_samples()
    assert isinstance(df, pd.DataFrame)

    # Check the default index values.
    assert df.index.names == (
        "country",
        "admin1_iso",
        "admin1_name",
        "admin2_name",
        "year",
    )


@pytest.mark.parametrize(
    "basemap", ["satellite", None, ipyleaflet.basemaps.OpenTopoMap]
)
@parametrize_with_cases("fixture,api", cases=".")
def test_plot_samples_interactive_map(fixture, api, basemap):
    m = api.plot_samples_interactive_map(basemap=basemap)
    assert isinstance(m, ipyleaflet.Map)


@parametrize_with_cases("fixture,api", cases=".")
def test_wgs_data_catalog(fixture, api):
    for rec in api.sample_sets()[["sample_set", "sample_count"]].itertuples():
        df = api.wgs_data_catalog(sample_set=rec.sample_set)
        assert isinstance(df, pd.DataFrame)
        expected_cols = [
            "sample_id",
            "alignments_bam",
            "snp_genotypes_vcf",
            "snp_genotypes_zarr",
        ]
        assert df.columns.to_list() == expected_cols
        assert len(df) == rec.sample_count
