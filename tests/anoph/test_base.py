import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.base import AnophelesBase


@pytest.fixture
def ag3_api(ag3_fixture):
    return AnophelesBase(
        url=ag3_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        gcs_url=_ag3.GCS_URL,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
    )


@pytest.fixture
def af1_api(af1_fixture):
    return AnophelesBase(
        url=af1_fixture.url,
        config_path=_af1.CONFIG_PATH,
        gcs_url=_af1.GCS_URL,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
    )


# N.B., here we use pytest_cases to parametrize tests. Each
# function whose name begins with "case_" defines a set of
# inputs to the test functions. See the documentation for
# pytest_cases for more information, e.g.:
#
# https://smarie.github.io/python-pytest-cases/#basic-usage
#
# We use this approach here because we want to use fixtures
# as test parameters, which is otherwise hard to do with
# pytest alone.


def case_ag3(ag3_fixture, ag3_api):
    return ag3_fixture, ag3_api


def case_af1(af1_fixture, af1_api):
    return af1_fixture, af1_api


@parametrize_with_cases("fixture,api", cases=".")
def test_config(fixture, api):
    config = api.config
    assert isinstance(config, dict)
    assert config == fixture.config


@parametrize_with_cases("fixture,api", cases=".")
def test_releases(fixture, api):
    releases = api.releases
    assert isinstance(releases, tuple)
    assert len(releases) > 0
    assert all([isinstance(r, str) for r in releases])
    assert releases == fixture.releases


@parametrize_with_cases("fixture,api", cases=".")
def test_client_location(fixture, api):
    location = api.client_location
    assert isinstance(location, str)


@parametrize_with_cases("fixture,api", cases=".")
def test_sample_sets_default(fixture, api):
    df = api.sample_sets()
    releases = api.releases
    expected = pd.concat(
        [fixture.release_manifests[release] for release in releases],
        axis=0,
        ignore_index=True,
    )
    expected.reset_index(inplace=True, drop=True)
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == ["sample_set", "sample_count", "release"]
    assert len(df) > 0
    assert_frame_equal(df[["sample_set", "sample_count"]], expected)


@parametrize_with_cases("fixture,api", cases=".")
def test_sample_sets_release(fixture, api):
    releases = api.releases
    for release in releases:
        df_ss = api.sample_sets(release=release)
        assert isinstance(df_ss, pd.DataFrame)
        assert df_ss.columns.tolist() == ["sample_set", "sample_count", "release"]
        assert len(df_ss) > 0
        expected = fixture.release_manifests[release]
        assert_frame_equal(df_ss[["sample_set", "sample_count"]], expected)
        assert (df_ss["release"] == release).all()


@parametrize_with_cases("fixture,api", cases=".")
def test_lookup_release(fixture, api):
    releases = api.releases
    for release in releases:
        df_ss = api.sample_sets(release=release)
        for s in df_ss["sample_set"]:
            assert api.lookup_release(s) == release
