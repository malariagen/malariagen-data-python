import pytest

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.sample_data import AnophelesSampleData


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesSampleData(
        url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        gcs_url=_ag3.GCS_URL,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesSampleData(
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


def validate_general_metadata(df):
    general_metadata_column_names = [
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
        "quarter",
    ]

    general_metadata_column_types = {
        "sample_id": object,
        "partner_sample_id": object,
        "contributor": object,
        "country": object,
        "location": object,
        "year": "int",
        "month": int,
        "latitude": float,
        "longitude": float,
        "sex_call": object,
        "sample_set": object,
        "release": object,
        "quarter": int,
    }

    # Check column names.
    assert df.columns.to_list() == general_metadata_column_names

    # Check column types.
    for c in df.columns:
        assert df[c].dtype == general_metadata_column_types[c]


@pytest.mark.parametrize("sample_set", ["AG1000G-AO", "AG1000G-BF-A"])
def test_general_metadata__single_sample_set(ag3_sim_api, sample_set):
    df = ag3_sim_api.general_metadata(sample_sets=sample_set)
    validate_general_metadata(df)

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sample_count.loc[sample_set]
    assert len(df) == expected_len


def test_general_metadata__multiple_sample_sets(ag3_sim_api):
    sample_sets = ["AG1000G-AO", "AG1000G-BF-A"]
    df = ag3_sim_api.general_metadata(sample_sets=sample_sets)
    validate_general_metadata(df)

    # Check number of rows.
    sample_count = ag3_sim_api.sample_sets().set_index("sample_set")["sample_count"]
    expected_len = sum([sample_count.loc[s] for s in sample_sets])
    assert len(df) == expected_len


def test_general_metadata__release(ag3_sim_api):
    release = "3.0"
    df = ag3_sim_api.general_metadata(sample_sets=release)
    validate_general_metadata(df)

    # Check number of rows.
    expected_len = ag3_sim_api.sample_sets(release=release)["sample_count"].sum()
    assert len(df) == expected_len
