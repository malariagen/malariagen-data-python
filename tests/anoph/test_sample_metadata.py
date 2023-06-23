import random

import ipyleaflet
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from pandas.testing import assert_frame_equal
from pytest_cases import parametrize_with_cases
from typeguard import suppress_type_checks

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


@pytest.fixture
def missing_metadata_api(fixture_dir):
    # In this fixture, one of the sample sets (AG1000G-BF-A) has missing files
    # for both AIM and cohorts metadata.
    return AnophelesSampleMetadata(
        url=(fixture_dir / "missing_metadata").as_uri(),
        config_path="config.json",
        gcs_url=None,
        major_version_number=3,
        major_version_path="v3",
        pre=False,
        aim_metadata_dtype={
            "aim_species_fraction_arab": "float64",
            "aim_species_fraction_colu": "float64",
            "aim_species_fraction_colu_no2l": "float64",
            "aim_species_gambcolu_arabiensis": object,
            "aim_species_gambiae_coluzzii": object,
            "aim_species": object,
        },
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


@parametrize_with_cases("fixture,api", cases=".")
def test_general_metadata_with_single_sample_set(fixture, api: AnophelesSampleMetadata):
    # Set up test.
    df_sample_sets = api.sample_sets().set_index("sample_set")
    sample_count = df_sample_sets["sample_count"]
    all_sample_sets = df_sample_sets.index.to_list()
    sample_set = random.choice(all_sample_sets)

    # Call function to be tested.
    df = api.general_metadata(sample_sets=sample_set)

    # Check output.
    validate_metadata(df, general_metadata_expected_columns())
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


@parametrize_with_cases("fixture,api", cases=".")
def test_general_metadata_with_multiple_sample_sets(
    fixture, api: AnophelesSampleMetadata
):
    # Set up the test.
    df_sample_sets = api.sample_sets().set_index("sample_set")
    sample_count = df_sample_sets["sample_count"]
    all_sample_sets = df_sample_sets.index.to_list()
    sample_sets = random.sample(all_sample_sets, 2)

    # Call function to be tested.
    df = api.general_metadata(sample_sets=sample_sets)

    # Check output.
    validate_metadata(df, general_metadata_expected_columns())
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


@parametrize_with_cases("fixture,api", cases=".")
def test_general_metadata_with_release(fixture, api: AnophelesSampleMetadata):
    # Set up the test.
    release = random.choice(api.releases)

    # Call function to be tested.
    df = api.general_metadata(sample_sets=release)

    # Check output.
    validate_metadata(df, general_metadata_expected_columns())
    expected_len = api.sample_sets(release=release)["sample_count"].sum()
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


def test_aim_metadata_with_single_sample_set(ag3_sim_api):
    # N.B., only Ag3 has AIM data.

    # Set up the test.
    df_sample_sets = ag3_sim_api.sample_sets().set_index("sample_set")
    sample_count = df_sample_sets["sample_count"]
    all_sample_sets = df_sample_sets.index.to_list()
    sample_set = random.choice(all_sample_sets)

    # Call function to be tested.
    df = ag3_sim_api.aim_metadata(sample_sets=sample_set)

    # Check output.
    validate_aim_metadata(df)
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


def test_aim_metadata_with_multiple_sample_sets(ag3_sim_api):
    # N.B., only Ag3 has AIM data.

    # Set up the test.
    df_sample_sets = ag3_sim_api.sample_sets().set_index("sample_set")
    sample_count = df_sample_sets["sample_count"]
    all_sample_sets = df_sample_sets.index.to_list()
    sample_sets = random.sample(all_sample_sets, 2)

    # Call function to be tested.
    df = ag3_sim_api.aim_metadata(sample_sets=sample_sets)

    # Check output.
    validate_aim_metadata(df)
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_aim_metadata_with_release(ag3_sim_api):
    # N.B., only Ag3 has AIM data.

    # Set up the test.
    release = random.choice(ag3_sim_api.releases)

    # Call function to be tested.
    df = ag3_sim_api.aim_metadata(sample_sets=release)

    # Check output.
    validate_aim_metadata(df)
    expected_len = ag3_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def test_aim_metadata_with_missing_file(
    missing_metadata_api: AnophelesSampleMetadata,
):
    # In this test, one of the sample sets (AG1000G-BF-A) has a missing file.
    # We expect this to be filled with empty values.
    api = missing_metadata_api

    # Set up test.
    df_sample_sets = api.sample_sets().set_index("sample_set")
    sample_count = df_sample_sets["sample_count"]
    all_sample_sets = df_sample_sets.index.to_list()

    for sample_set in all_sample_sets:
        # Call function to be tested.
        df = api.aim_metadata(sample_sets=sample_set)

        # Check output.
        validate_aim_metadata(df)
        expected_len = sample_count.loc[sample_set]
        assert len(df) == expected_len


def cohorts_metadata_expected_columns(has_cohorts_by_quarter):
    if has_cohorts_by_quarter:
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


def validate_cohorts_metadata(df, has_cohorts_by_quarter):
    # N.B., older cohorts metadata only has cohorts by year and month.
    # Newer cohorts metadata also has cohorts by quarter.
    expected_columns = cohorts_metadata_expected_columns(
        has_cohorts_by_quarter=has_cohorts_by_quarter
    )
    validate_metadata(df, expected_columns)


@parametrize_with_cases("fixture,api", cases=".")
def test_cohorts_metadata_with_single_sample_set(fixture, api: AnophelesSampleMetadata):
    # Set up test.
    df_sample_sets = api.sample_sets().set_index("sample_set")
    sample_count = df_sample_sets["sample_count"]
    all_sample_sets = df_sample_sets.index.to_list()
    sample_set = random.choice(all_sample_sets)

    # Call function to be tested.
    df = api.cohorts_metadata(sample_sets=sample_set)

    # Check output.
    validate_cohorts_metadata(df, has_cohorts_by_quarter=fixture.has_cohorts_by_quarter)
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


@parametrize_with_cases("fixture,api", cases=".")
def test_cohorts_metadata_with_multiple_sample_sets(
    fixture, api: AnophelesSampleMetadata
):
    # Set up test.
    df_sample_sets = api.sample_sets().set_index("sample_set")
    sample_count = df_sample_sets["sample_count"]
    all_sample_sets = df_sample_sets.index.to_list()
    sample_sets = random.sample(all_sample_sets, 2)

    # Call function to be tested.
    df = api.cohorts_metadata(sample_sets=sample_sets)

    # Check output.
    validate_cohorts_metadata(df, has_cohorts_by_quarter=fixture.has_cohorts_by_quarter)
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


@parametrize_with_cases("fixture,api", cases=".")
def test_cohorts_metadata_with_release(fixture, api: AnophelesSampleMetadata):
    # Set up test.
    release = random.choice(api.releases)

    # Call function to be tested.
    df = api.cohorts_metadata(sample_sets=release)

    # Check output.
    validate_cohorts_metadata(df, has_cohorts_by_quarter=fixture.has_cohorts_by_quarter)
    expected_len = api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


def test_cohorts_metadata_with_missing_file(
    missing_metadata_api: AnophelesSampleMetadata,
):
    # In this test, one of the sample sets (AG1000G-BF-A) has a missing file.
    # We expect this to be filled with empty values.
    api = missing_metadata_api

    # Set up test.
    df_sample_sets = api.sample_sets().set_index("sample_set")
    sample_count = df_sample_sets["sample_count"]
    all_sample_sets = df_sample_sets.index.to_list()

    for sample_set in all_sample_sets:
        # Call function to be tested.
        df = api.cohorts_metadata(sample_sets=sample_set)

        # Check output.
        validate_cohorts_metadata(df, has_cohorts_by_quarter=True)
        expected_len = sample_count.loc[sample_set]
        assert len(df) == expected_len


def sample_metadata_expected_columns(has_aims, has_cohorts_by_quarter):
    expected_columns = general_metadata_expected_columns()
    if has_aims:
        expected_columns.update(aim_metadata_expected_columns())
    expected_columns.update(
        cohorts_metadata_expected_columns(has_cohorts_by_quarter=has_cohorts_by_quarter)
    )
    return expected_columns


@parametrize_with_cases("fixture,api", cases=".")
def test_sample_metadata_default(fixture, api: AnophelesSampleMetadata):
    # Default is all releases.
    df_default = api.sample_metadata()
    df_all = api.sample_metadata(sample_sets=api.releases)
    assert_frame_equal(df_default, df_all)


@parametrize_with_cases("fixture,api", cases=".")
def test_sample_metadata_with_single_sample_set(fixture, api: AnophelesSampleMetadata):
    # Set up test.
    df_sample_sets = api.sample_sets().set_index("sample_set")
    sample_count = df_sample_sets["sample_count"]
    all_sample_sets = df_sample_sets.index.to_list()
    sample_set = random.choice(all_sample_sets)

    # Call function to be tested.
    df = api.sample_metadata(sample_sets=sample_set)

    # Check output.
    validate_metadata(
        df,
        sample_metadata_expected_columns(
            has_aims=fixture.has_aims,
            has_cohorts_by_quarter=fixture.has_cohorts_by_quarter,
        ),
    )
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


@parametrize_with_cases("fixture,api", cases=".")
def test_sample_metadata_with_multiple_sample_sets(
    fixture, api: AnophelesSampleMetadata
):
    # Set up test.
    df_sample_sets = api.sample_sets().set_index("sample_set")
    sample_count = df_sample_sets["sample_count"]
    all_sample_sets = df_sample_sets.index.to_list()
    sample_sets = random.sample(all_sample_sets, 2)

    # Call function to be tested.
    df = api.sample_metadata(sample_sets=sample_sets)

    # Check output.
    validate_metadata(
        df,
        sample_metadata_expected_columns(
            has_aims=fixture.has_aims,
            has_cohorts_by_quarter=fixture.has_cohorts_by_quarter,
        ),
    )
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


@parametrize_with_cases("fixture,api", cases=".")
def test_sample_metadata_with_release(fixture, api: AnophelesSampleMetadata):
    # Set up test.
    release = random.choice(api.releases)

    # Call function to be tested.
    df = api.sample_metadata(sample_sets=release)

    # Check output.
    validate_metadata(
        df,
        sample_metadata_expected_columns(
            has_aims=fixture.has_aims,
            has_cohorts_by_quarter=fixture.has_cohorts_by_quarter,
        ),
    )
    expected_len = api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len


@parametrize_with_cases("fixture,api", cases=".")
def test_sample_metadata_with_duplicate_sample_sets(
    fixture, api: AnophelesSampleMetadata
):
    # Set up test.
    release = random.choice(api.releases)
    df_sample_sets = api.sample_sets(release=release).set_index("sample_set")
    all_sample_sets = df_sample_sets.index.to_list()
    sample_set = random.choice(all_sample_sets)

    # Call function to be tested.
    assert_frame_equal(
        api.sample_metadata(sample_sets=[sample_set, sample_set]),
        api.sample_metadata(sample_sets=sample_set),
    )
    assert_frame_equal(
        api.sample_metadata(sample_sets=[release, release]),
        api.sample_metadata(sample_sets=release),
    )
    assert_frame_equal(
        api.sample_metadata(sample_sets=[release, sample_set]),
        api.sample_metadata(sample_sets=release),
    )


def test_sample_metadata_with_query(ag3_sim_api):
    df = ag3_sim_api.sample_metadata(sample_query="country == 'Burkina Faso'")
    validate_metadata(
        df, sample_metadata_expected_columns(has_aims=True, has_cohorts_by_quarter=True)
    )
    assert (df["country"] == "Burkina Faso").all()


def test_sample_metadata_with_indices(ag3_sim_api):
    df_all = ag3_sim_api.sample_metadata()
    query = "country == 'Burkina Faso'"
    indices = np.nonzero(df_all.eval(query))[0].tolist()
    df1 = ag3_sim_api.sample_metadata(sample_query=query)
    df2 = ag3_sim_api.sample_metadata(sample_indices=indices)
    validate_metadata(
        df1,
        sample_metadata_expected_columns(has_aims=True, has_cohorts_by_quarter=True),
    )
    assert (df1["country"] == "Burkina Faso").all()
    validate_metadata(
        df2,
        sample_metadata_expected_columns(has_aims=True, has_cohorts_by_quarter=True),
    )
    assert (df2["country"] == "Burkina Faso").all()
    assert_frame_equal(df1, df2)


@parametrize_with_cases("fixture,api", cases=".")
def test_sample_metadata_quarter(fixture, api: AnophelesSampleMetadata):
    df = api.sample_metadata()

    # Check that quarter only contains the expected values
    expected_quarter_values = {-1, 1, 2, 3, 4}
    assert df["quarter"].isin(expected_quarter_values).all()

    # Check that quarter is -1 when month is -1
    assert np.all(df.query("month == -1")["quarter"] == -1)

    # Check that quarter is derived from month, in cases where it is not -1
    assert (df.query("month == -1")["quarter"] == -1).all()
    assert (df.query("month in [1, 2, 3]")["quarter"] == 1).all()
    assert (df.query("month in [4, 5, 6]")["quarter"] == 2).all()
    assert (df.query("month in [7, 8, 9]")["quarter"] == 3).all()
    assert (df.query("month in [10, 11, 12]")["quarter"] == 4).all()


def test_sample_metadata_with_missing_file(
    missing_metadata_api: AnophelesSampleMetadata,
):
    # In this test, one of the sample sets (AG1000G-BF-A) has a missing file.
    # We expect this to be filled with empty values.
    api = missing_metadata_api

    # Set up test.
    df_sample_sets = api.sample_sets().set_index("sample_set")
    sample_count = df_sample_sets["sample_count"]
    all_sample_sets = df_sample_sets.index.to_list()

    for sample_set in all_sample_sets:
        # Call function to be tested.
        df = api.sample_metadata(sample_sets=sample_set)

        # Check output.
        validate_metadata(
            df,
            sample_metadata_expected_columns(
                has_aims=True, has_cohorts_by_quarter=True
            ),
        )
        expected_len = sample_count.loc[sample_set]
        assert len(df) == expected_len


@parametrize_with_cases("fixture,api", cases=".")
def test_extra_metadata_errors(fixture, api):
    # Bad type.
    with suppress_type_checks():
        with pytest.raises(TypeError):
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
    # Set up test.
    df_sample_sets = api.sample_sets().set_index("sample_set")
    sample_count = df_sample_sets["sample_count"]
    all_sample_sets = df_sample_sets.index.to_list()
    sample_set = random.choice(all_sample_sets)

    # Call function to be tested.
    df = api.wgs_data_catalog(sample_set=sample_set)

    # Check output.
    assert isinstance(df, pd.DataFrame)
    expected_cols = [
        "sample_id",
        "alignments_bam",
        "snp_genotypes_vcf",
        "snp_genotypes_zarr",
    ]
    assert df.columns.to_list() == expected_cols
    assert len(df) == sample_count.loc[sample_set]


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_samples_bar(fixture, api):
    # By country.
    fig = api.plot_samples_bar(
        x="country",
        show=False,
    )
    assert isinstance(fig, go.Figure)

    # By year.
    fig = api.plot_samples_bar(
        x="year",
        show=False,
    )
    assert isinstance(fig, go.Figure)

    # By country and taxon.
    fig = api.plot_samples_bar(
        x="country",
        color="taxon",
        show=False,
    )
    assert isinstance(fig, go.Figure)

    # By year and country.
    fig = api.plot_samples_bar(
        x="year",
        color="country",
        show=False,
    )
    assert isinstance(fig, go.Figure)

    # Not sorted.
    fig = api.plot_samples_bar(
        x="country",
        color="taxon",
        sort=False,
        show=False,
    )
    assert isinstance(fig, go.Figure)
