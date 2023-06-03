import pytest
import xarray as xr

from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.aim_data import AnophelesAimData


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesAimData(
        url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        gcs_url=_ag3.GCS_URL,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
        aim_metadata_dtype={
            "aim_species_fraction_arab": "float64",
            "aim_species_fraction_colu": "float64",
            "aim_species_fraction_colu_no2l": "float64",
            "aim_species_gambcolu_arabiensis": object,
            "aim_species_gambiae_coluzzii": object,
            "aim_species": object,
        },
        aim_ids=("gambcolu_vs_arab", "gamb_vs_colu"),
        gff_gene_type="gene",
        gff_default_attributes=("ID", "Parent", "Name", "description"),
    )


@pytest.mark.parametrize("aims", ["gambcolu_vs_arab", "gamb_vs_colu"])
def test_aim_variants(aims, ag3_sim_api):
    ds = ag3_sim_api.aim_variants(aims=aims)
    assert isinstance(ds, xr.Dataset)
    # TODO more tests
