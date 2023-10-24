import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.cohort_group_metadata import AnophelesCohortGroupMetadata


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesCohortGroupMetadata(
        url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        gcs_url=_ag3.GCS_URL,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesCohortGroupMetadata(
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


def cohort_group_metadata_expected_columns():
    return {
        "cohort_id": "O",
        "cohort_size": "i",
        "country": "O",
        "country_alpha2": "O",
        "country_alpha3": "O",
        "taxon": "O",
        "year": "i",
        "quarter": "i",
        "month": "i",
        "admin1_name": "O",
        "admin1_iso": "O",
        "admin1_geoboundaries_shape_id": "O",
        "admin1_representative_longitude": "f",
        "admin1_representative_latitude": "f",
    }


def validate_cohort_group_metadata(df, expected_columns):
    # Check column names.
    expected_column_names = list(expected_columns.keys())
    assert df.columns.to_list() == expected_column_names

    # Check column types.
    for c in df.columns:
        assert df[c].dtype.kind == expected_columns[c]


@parametrize_with_cases("fixture,api", cases=".")
def test_cohort_group_metadata(fixture, api: AnophelesCohortGroupMetadata):
    # Set up the test.
    cohort_name = "admin1_month"
    # Call function to be tested.
    df_cohorts = api.cohort_group_metadata(cohort_name)
    # Check output.
    validate_cohort_group_metadata(df_cohorts, cohort_group_metadata_expected_columns())
    # # Check values against cohort metadata
    # df_default = api.sample_metadata()
    # assert df_cohorts['cohort_id'] in df_default['cohort_admin1_month']


@parametrize_with_cases("fixture,api", cases=".")
def test_cohort_group_metadata_with_query(fixture, api: AnophelesCohortGroupMetadata):
    cohort_name = "admin1_month"
    df_cohorts = api.cohort_group_metadata(
        cohort_name, cohort_group_query="country == 'Burkina Faso'"
    )
    validate_cohort_group_metadata(df_cohorts, cohort_group_metadata_expected_columns())
    assert (df_cohorts["country"] == "Burkina Faso").all()
