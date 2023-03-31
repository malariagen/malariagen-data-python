import json
from pathlib import Path

import pandas as pd
import pytest

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.base import AnophelesBase

# We are going to create some data locally which follows
# the same layout and format of the real data in GCS,
# but which is much smaller and so can be used for faster
# test runs. This data is referred to here as the
# "data fixture".


class Fixture:
    # This is the location where the data fixture will be stored.
    root_path = Path(__file__).parent.resolve() / "fixture"

    # These parameters are for Ag3-style data fixture.
    ag_bucket = "vo_agam_release"
    ag_path = (root_path / ag_bucket).resolve()
    ag_path.mkdir(parents=True, exist_ok=True)
    ag_url = ag_path.as_uri()

    # These parameters are for Af1-style data fixture.
    af_bucket = "vo_afun_release"
    af_path = (root_path / af_bucket).resolve()
    af_path.mkdir(parents=True, exist_ok=True)
    af_url = af_path.as_uri()

    @staticmethod
    @pytest.fixture(scope="session", autouse=True)
    def create_ag3_config(force: bool = False):
        # Here we create the release config file for an Ag3-style
        # data fixture.
        config_path = Fixture.ag_path / _ag3.CONFIG_PATH
        if force or not config_path.exists():
            config = {
                "PUBLIC_RELEASES": ["3.0"],
            }
            with config_path.open(mode="w") as f:
                json.dump(config, f)

    @staticmethod
    @pytest.fixture(scope="session", autouse=True)
    def create_af1_config(force: bool = False):
        # Here we create the release config file for an Af1-style
        # data fixture.
        config_path = Fixture.af_path / _af1.CONFIG_PATH
        if force or not config_path.exists():
            config = {
                "PUBLIC_RELEASES": ["1.0"],
            }
            with config_path.open(mode="w") as f:
                json.dump(config, f)

    @staticmethod
    @pytest.fixture(scope="session", autouse=True)
    def create_ag3_public_release_manifest(force: bool = False):
        # Here we create a release manifest for an Ag3-style
        # public release. Note this is not the exact same data
        # as the real release.
        release_path = Fixture.ag_path / _ag3.MAJOR_VERSION_GCS_STR
        release_path.mkdir(parents=True, exist_ok=True)
        manifest_path = release_path / "manifest.tsv"
        if force or not manifest_path.exists():
            manifest = pd.DataFrame(
                {
                    "sample_set": ["AG1000G-AO", "AG1000G-BF-A"],
                    "sample_count": [81, 181],
                }
            )
            manifest.to_csv(manifest_path, index=False, sep="\t")

    @staticmethod
    @pytest.fixture(scope="session", autouse=True)
    def create_ag3_pre_release_manifest(force: bool = False):
        # Here we create a release manifest for an Ag3-style
        # pre-release. Note this is not the exact same data
        # as the real release.
        release_path = Fixture.ag_path / "v3.1"
        release_path.mkdir(parents=True, exist_ok=True)
        manifest_path = release_path / "manifest.tsv"
        if force or not manifest_path.exists():
            manifest = pd.DataFrame(
                {
                    "sample_set": [
                        "1177-VO-ML-LEHMANN-VMF00015",
                        "1237-VO-BJ-DJOGBENOU-VMF00050",
                    ],
                    "sample_count": [23, 90],
                }
            )
            manifest.to_csv(manifest_path, index=False, sep="\t")

    @staticmethod
    @pytest.fixture(scope="session", autouse=True)
    def create_af1_public_release_manifest(force: bool = False):
        # Here we create a release manifest for an Af1-style
        # public release. Note this is not the exact same data
        # as the real release.
        release_path = Fixture.af_path / _af1.MAJOR_VERSION_GCS_STR
        release_path.mkdir(parents=True, exist_ok=True)
        manifest_path = release_path / "manifest.tsv"
        if force or not manifest_path.exists():
            manifest = pd.DataFrame(
                {
                    "sample_set": [
                        "1229-VO-GH-DADZIE-VMF00095",
                        "1230-VO-GA-CF-AYALA-VMF00045",
                        "1231-VO-MULTI-WONDJI-VMF00043",
                    ],
                    "sample_count": [36, 50, 320],
                }
            )
            manifest.to_csv(manifest_path, index=False, sep="\t")


# N.B., we want to test behaviour of the AnophelesBase class
# both with and without pre-releases.

# Set up using Ag3-style data fixture and pre-releases.
ag3_api = AnophelesBase(
    url=Fixture.ag_url,
    config_path=_ag3.CONFIG_PATH,
    gcs_url=_ag3.GCS_URL,
    major_version_number=_ag3.MAJOR_VERSION_INT,
    major_version_path=_ag3.MAJOR_VERSION_GCS_STR,
    pre=True,
)
ag3_param = pytest.param(ag3_api, id="ag3")

# Set up using the Af1-style data fixture and public releases only.
af1_api = AnophelesBase(
    url=Fixture.af_url,
    config_path=_af1.CONFIG_PATH,
    gcs_url=_af1.GCS_URL,
    major_version_number=_af1.MAJOR_VERSION_INT,
    major_version_path=_af1.MAJOR_VERSION_GCS_STR,
    pre=False,
)
af1_param = pytest.param(af1_api, id="af1")

# Run tests for Ag3-style and Af1-style data fixtures.
api_params = [ag3_param, af1_param]


@pytest.mark.parametrize("api", api_params)
def test_config(api):
    config = api.config
    assert isinstance(config, dict)


@pytest.mark.parametrize("api", api_params)
def test_releases(api):
    releases = api.releases
    assert isinstance(releases, tuple)
    assert len(releases) > 0
    assert all([isinstance(r, str) for r in releases])


@pytest.mark.parametrize("api", api_params)
def test_api_location(api):
    location = api.client_location
    assert isinstance(location, str)


@pytest.mark.parametrize("api", api_params)
def test_sample_sets_default(api):
    df = api.sample_sets()
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == ["sample_set", "sample_count", "release"]
    assert len(df) > 0


@pytest.mark.parametrize("api", api_params)
def test_sample_sets_specific(api):
    releases = api.releases
    for release in releases:
        df = api.sample_sets(release=release)
        assert isinstance(df, pd.DataFrame)
        assert df.columns.tolist() == ["sample_set", "sample_count", "release"]
        assert len(df) > 0


@pytest.mark.parametrize("api", api_params)
def test_lookup_release(api):
    releases = api.releases
    for release in releases:
        df_ss = api.sample_sets(release=release)
        for s in df_ss["sample_set"]:
            assert api.lookup_release(s) == release
