import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.cohort_data import AnophelesCohortData


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesCohortData(
        url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        gcs_url=_ag3.GCS_URL,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
        cohorts_analysis=None,
    )


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def cohort_data_expected_columns():
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


def validate_cohort_data(df, expected_columns):
    # Check column names.
    expected_column_names = list(expected_columns.keys())
    assert df.columns.to_list() == expected_column_names

    # Check column types.
    for c in df.columns:
        assert df[c].dtype.kind == expected_columns[c]


@parametrize_with_cases("fixture,api", cases=".")
def test_cohort_data(fixture, api: AnophelesCohortData):
    # Set up the test.
    cohort_name = "admin1_month"
    # Call function to be tested.
    df_cohorts = api.cohorts(cohort_name)
    # Check output.
    validate_cohort_data(df_cohorts, cohort_data_expected_columns())
