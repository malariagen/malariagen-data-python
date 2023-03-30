import sys
from pathlib import Path

import pytest

from malariagen_data.anoph.base import AnophelesBase

fixture_dir = Path(__file__).parent.resolve() / "fixture"


ag3_url = (fixture_dir / "vo_agam_release").resolve().as_uri()
ag3_config_path = "v3-config.json"
ag3_gcs_url = "gs://vo_agam_release/"

ag3_public = AnophelesBase(
    url=ag3_url,
    config_path=ag3_config_path,
    bokeh_output_notebook=False,
    log=sys.stderr,
    debug=False,
    show_progress=False,
    check_location=False,
    pre=False,
    gcs_url=ag3_gcs_url,
    major_version_number=3,
    major_version_path="v3",
)

ag3_pre = AnophelesBase(
    url=ag3_url,
    config_path=ag3_config_path,
    bokeh_output_notebook=False,
    log=sys.stderr,
    debug=False,
    show_progress=False,
    check_location=False,
    pre=True,
    gcs_url=ag3_gcs_url,
    major_version_number=3,
    major_version_path="v3",
)


@pytest.mark.parametrize("client", [ag3_public, ag3_pre])
def test_config(client):
    config = client.config
    assert isinstance(config, dict)


@pytest.mark.parametrize("client", [ag3_public, ag3_pre])
def test_releases(client):
    releases = client.releases
    assert isinstance(releases, tuple)
    assert len(releases) > 0
    assert all([isinstance(r, str) for r in releases])


@pytest.mark.parametrize("client", [ag3_public, ag3_pre])
def test_client_location(client):
    location = client.client_location
    assert isinstance(location, str)
