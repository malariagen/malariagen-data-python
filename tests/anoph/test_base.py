import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from malariagen_data.anoph.base import AnophelesBase

# We are going to create some data locally which follows
# the same layout and format of the real data in GCS,
# but which is much smaller and so can be used for faster
# test runs. This data is referred to here as the
# "data fixture".

# This is the location where the data fixture will be stored.
fixture_path = Path(__file__).parent.resolve() / "fixture"

# These parameters are for Ag3-style data fixture.
ag_bucket_name = "vo_agam_release"
ag_gcs_url = f"gs://{ag_bucket_name}/"
ag_fixture_path = (fixture_path / ag_bucket_name).resolve()
ag_fixture_path.mkdir(parents=True, exist_ok=True)
ag_fixture_url = ag_fixture_path.as_uri()
ag3_config_path = "v3-config.json"

# These parameters are for Af1-style data fixture.
af_bucket_name = "vo_afun_release"
af_gcs_url = f"gs://{af_bucket_name}/"
af_fixture_path = (fixture_path / af_bucket_name).resolve()
af_fixture_path.mkdir(parents=True, exist_ok=True)
af_fixture_url = af_fixture_path.as_uri()
af1_config_path = "v1.0-config.json"


@pytest.fixture(scope="session", autouse=True)
def ag3_fixture_config(force: bool = False):
    # Here we create the release config file for an Ag3-style
    # data fixture.
    config_path = ag_fixture_path / ag3_config_path
    if force or not config_path.exists():
        config = {
            "PUBLIC_RELEASES": ["3.0"],
        }
        with config_path.open(mode="w") as f:
            json.dump(config, f)


@pytest.fixture(scope="session", autouse=True)
def af1_fixture_config(force: bool = False):
    # Here we create the release config file for an Af1-style
    # data fixture.
    config_path = af_fixture_path / af1_config_path
    if force or not config_path.exists():
        config = {
            "PUBLIC_RELEASES": ["1.0"],
        }
        with config_path.open(mode="w") as f:
            json.dump(config, f)


@pytest.fixture(scope="session", autouse=True)
def ag3_fixture_public_release_manifest(force: bool = False):
    # Here we create a release manifest for an Ag3-style
    # public release. Note this is not the exact same data
    # as the real release.
    public_release_path = ag_fixture_path / "v3"
    public_release_path.mkdir(parents=True, exist_ok=True)
    public_release_manifest_path = public_release_path / "manifest.tsv"
    if force or not public_release_manifest_path.exists():
        public_release_manifest = pd.DataFrame(
            {
                "sample_set": ["AG1000G-AO", "AG1000G-BF-A"],
                "sample_count": [81, 181],
            }
        )
        public_release_manifest.to_csv(
            public_release_manifest_path, index=False, sep="\t"
        )


@pytest.fixture(scope="session", autouse=True)
def af1_fixture_public_release_manifest(force: bool = False):
    # Here we create a release manifest for an Af1-style
    # public release. Note this is not the exact same data
    # as the real release.
    public_release_path = af_fixture_path / "v1.0"
    public_release_path.mkdir(parents=True, exist_ok=True)
    public_release_manifest_path = public_release_path / "manifest.tsv"
    if force or not public_release_manifest_path.exists():
        public_release_manifest = pd.DataFrame(
            {
                "sample_set": [
                    "1229-VO-GH-DADZIE-VMF00095",
                    "1230-VO-GA-CF-AYALA-VMF00045",
                    "1231-VO-MULTI-WONDJI-VMF00043",
                ],
                "sample_count": [36, 50, 320],
            }
        )
        public_release_manifest.to_csv(
            public_release_manifest_path, index=False, sep="\t"
        )


@pytest.fixture(scope="session", autouse=True)
def ag3_fixture_pre_release_manifest(force: bool = False):
    # Here we create a release manifest for an Ag3-style
    # pre-release. Note this is not the exact same data
    # as the real release.
    pre_release_path = ag_fixture_path / "v3.1"
    pre_release_path.mkdir(parents=True, exist_ok=True)
    pre_release_manifest_path = pre_release_path / "manifest.tsv"
    if force or not pre_release_manifest_path.exists():
        pre_release_manifest = pd.DataFrame(
            {
                "sample_set": [
                    "1177-VO-ML-LEHMANN-VMF00015",
                    "1237-VO-BJ-DJOGBENOU-VMF00050",
                ],
                "sample_count": [23, 90],
            }
        )
        pre_release_manifest.to_csv(pre_release_manifest_path, index=False, sep="\t")


# N.B., we want to test behaviour of the AnophelesBase class
# both with and without pre-releases.

# Set up using the Ag3-style data fixture and public releases only.
ag3_public = AnophelesBase(
    url=ag_fixture_url,
    config_path=ag3_config_path,
    bokeh_output_notebook=False,
    log=sys.stderr,
    debug=True,
    show_progress=False,
    check_location=False,
    pre=False,
    gcs_url=ag_gcs_url,
    major_version_number=3,
    major_version_path="v3",
)
ag3_public_param = pytest.param(ag3_public, id="ag3_public")

# Set up using Ag3-style data fixture and pre-releases.
ag3_pre = AnophelesBase(
    url=ag_fixture_url,
    config_path=ag3_config_path,
    bokeh_output_notebook=False,
    log=sys.stderr,
    debug=True,
    show_progress=False,
    check_location=False,
    pre=True,
    gcs_url=ag_gcs_url,
    major_version_number=3,
    major_version_path="v3",
)
ag3_pre_param = pytest.param(ag3_pre, id="ag3_pre")

# Set up using the Af1-style data fixture and public releases only.
af1_public = AnophelesBase(
    url=af_fixture_url,
    config_path=af1_config_path,
    bokeh_output_notebook=False,
    log=sys.stderr,
    debug=True,
    show_progress=False,
    check_location=False,
    pre=False,
    gcs_url=af_gcs_url,
    major_version_number=1,
    major_version_path="v1.0",
)
af1_public_param = pytest.param(af1_public, id="af1_public")


@pytest.mark.parametrize("api", [ag3_public_param, ag3_pre_param, af1_public_param])
def test_config(api):
    config = api.config
    assert isinstance(config, dict)


@pytest.mark.parametrize("api", [ag3_public_param, ag3_pre_param, af1_public_param])
def test_releases(api):
    releases = api.releases
    assert isinstance(releases, tuple)
    assert len(releases) > 0
    assert all([isinstance(r, str) for r in releases])


@pytest.mark.parametrize("api", [ag3_public_param, ag3_pre_param, af1_public_param])
def test_api_location(api):
    location = api.client_location
    assert isinstance(location, str)


@pytest.mark.parametrize("api", [ag3_public_param, ag3_pre_param, af1_public_param])
def test_sample_sets_default(api):
    df = api.sample_sets()
    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == ["sample_set", "sample_count", "release"]
    assert len(df) > 0


@pytest.mark.parametrize("api", [ag3_public_param, ag3_pre_param, af1_public_param])
def test_sample_sets_specific(api):
    releases = api.releases
    for release in releases:
        df = api.sample_sets(release=release)
        assert isinstance(df, pd.DataFrame)
        assert df.columns.tolist() == ["sample_set", "sample_count", "release"]
        assert len(df) > 0


@pytest.mark.parametrize("api", [ag3_public_param, ag3_pre_param, af1_public_param])
def test_lookup_release(api):
    releases = api.releases
    for release in releases:
        df_ss = api.sample_sets(release=release)
        for s in df_ss["sample_set"]:
            assert api.lookup_release(s) == release
