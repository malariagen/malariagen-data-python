import pytest

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.genome_sequence import AnophelesGenomeSequenceData

# import zarr


@pytest.fixture(scope="session")
def ag3_api(ag3_fixture):
    return AnophelesGenomeSequenceData(
        url=ag3_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        gcs_url=_ag3.GCS_URL,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
    )


@pytest.fixture(scope="session")
def af1_api(af1_fixture):
    return AnophelesGenomeSequenceData(
        url=af1_fixture.url,
        config_path=_af1.CONFIG_PATH,
        gcs_url=_af1.GCS_URL,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=True,
    )


def test_contigs_ag3(ag3_api, ag3_fixture):
    contigs = ag3_api.contigs
    assert isinstance(contigs, tuple)
    assert all([isinstance(c, str) for c in contigs])
    assert contigs == tuple(ag3_fixture.config["CONTIGS"])


def test_open_genome_ag3(ag3_api):
    # TODO
    # root = api.open_genome()
    # assert isinstance(root, zarr.hierarchy.Group)
    pass
