import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.base import AnophelesBase


@pytest.fixture(scope="session")
def ag3_api(ag3_fixture):
    return AnophelesBase(
        url=ag3_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        gcs_url=_ag3.GCS_URL,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
    )


@pytest.fixture(scope="session")
def af1_api(af1_fixture):
    return AnophelesBase(
        url=af1_fixture.url,
        config_path=_af1.CONFIG_PATH,
        gcs_url=_af1.GCS_URL,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
    )


def test_config_ag3(ag3_api, ag3_fixture):
    config = ag3_api.config
    assert isinstance(config, dict)
    assert config == ag3_fixture.config


def test_releases_ag3(ag3_api, ag3_fixture):
    releases = ag3_api.releases
    assert isinstance(releases, tuple)
    assert len(releases) > 0
    assert all([isinstance(r, str) for r in releases])
    assert releases == ag3_fixture.releases


def test_client_location_ag3(ag3_api):
    location = ag3_api.client_location
    assert isinstance(location, str)


def test_sample_sets_default_ag3(ag3_api):
    df = ag3_api.sample_sets()
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == ["sample_set", "sample_count", "release"]
    assert len(df) > 0


def test_sample_sets_release_ag3(ag3_api, ag3_fixture):
    releases = ag3_api.releases
    for release in releases:
        df = ag3_api.sample_sets(release=release)
        assert isinstance(df, pd.DataFrame)
        assert df.columns.tolist() == ["sample_set", "sample_count", "release"]
        assert len(df) > 0
        manifest = ag3_fixture.release_manifests[release]
        assert_frame_equal(df[["sample_set", "sample_count"]], manifest)
        assert (df["release"] == release).all()


def test_lookup_release_ag3(ag3_api):
    releases = ag3_api.releases
    for release in releases:
        df_ss = ag3_api.sample_sets(release=release)
        for s in df_ss["sample_set"]:
            assert ag3_api.lookup_release(s) == release
