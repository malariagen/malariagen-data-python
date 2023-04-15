# import dask.array as da
# import numpy as np
import pytest

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.sample_data import AnophelesSampleData

# import zarr
# from pytest_cases import parametrize_with_cases


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


def test_general_metadata_sample_set(ag3_sim_fixture, ag3_sim_api):
    df = ag3_sim_api.general_metadata(sample_set="AG1000G-AO")
    assert df.columns.to_list() == [
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
    ]
    assert df["sample_id"].dtype == str
    # TODO check other column dtypes
