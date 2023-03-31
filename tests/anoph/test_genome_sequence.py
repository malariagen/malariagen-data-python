import pytest

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.genome_sequence import AnophelesGenomeSequenceData

from .test_base import Fixture as BaseFixture

# import zarr


class Fixture(BaseFixture):
    # TODO Create genome sequence fixture data.
    pass


# Set up using the Ag3-style data fixture and pre-releases.
ag3_api = AnophelesGenomeSequenceData(
    contigs=_ag3.CONTIGS,
    genome_fasta_path=_ag3.GENOME_FASTA_PATH,
    genome_fai_path=_ag3.GENOME_FAI_PATH,
    genome_zarr_path=_ag3.GENOME_ZARR_PATH,
    genome_ref_id=_ag3.GENOME_REF_ID,
    genome_ref_name=_ag3.GENOME_REF_NAME,
    # These parameters are passed through to AnophelesBase
    url=Fixture.ag_url,
    config_path=_ag3.CONFIG_PATH,
    gcs_url=_ag3.GCS_URL,
    major_version_number=_ag3.MAJOR_VERSION_INT,
    major_version_path=_ag3.MAJOR_VERSION_GCS_STR,
    pre=True,
)
ag3_param = pytest.param(ag3_api, id="ag3")

# Set up using the Af1-style data fixture and public releases only.
af1_api = AnophelesGenomeSequenceData(
    contigs=_af1.CONTIGS,
    genome_fasta_path=_af1.GENOME_FASTA_PATH,
    genome_fai_path=_af1.GENOME_FAI_PATH,
    genome_zarr_path=_af1.GENOME_ZARR_PATH,
    genome_ref_id=_af1.GENOME_REF_ID,
    genome_ref_name=_af1.GENOME_REF_NAME,
    # These parameters are passed through to AnophelesBase
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
def test_contigs(api):
    contigs = api.contigs
    assert isinstance(contigs, tuple)
    assert all([isinstance(c, str) for c in contigs])


@pytest.mark.parametrize("api", api_params)
def test_open_genome(api):
    # TODO
    # root = api.open_genome()
    # assert isinstance(root, zarr.hierarchy.Group)
    pass
