import itertools
import random

import plotly.graph_objects as go
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

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
        aim_palettes=_ag3.AIM_PALETTES,
        gff_gene_type="gene",
        gff_gene_name_attribute="Name",
        gff_default_attributes=("ID", "Parent", "Name", "description"),
    )


@pytest.mark.parametrize("aims", ["gambcolu_vs_arab", "gamb_vs_colu"])
def test_aim_variants(aims, ag3_sim_api):
    api = ag3_sim_api

    # Call function to be tested.
    ds = api.aim_variants(aims=aims)

    # Check return type.
    assert isinstance(ds, xr.Dataset)

    # Check data variables.
    expected_data_vars = {"variant_allele"}
    assert set(ds.data_vars) == expected_data_vars

    # Check coordinate variables.
    expected_coords = {"variant_contig", "variant_position"}
    assert set(ds.coords) == expected_coords

    # Check dimensions.
    expected_dims = {"variants", "alleles"}
    assert set(ds.dims) == expected_dims

    # Check variant_contig variable.
    x = ds["variant_contig"]
    assert x.dims == ("variants",)
    assert x.dtype == "uint8"

    # Check variant_position variable.
    x = ds["variant_position"]
    assert x.dims == ("variants",)
    assert x.dtype == "int64" or "int32"

    # Check variant_allele variable.
    x = ds["variant_allele"]
    assert x.dims == ("variants", "alleles")
    assert x.dtype == "S1"

    # Check attributes.
    assert tuple(ds.attrs["contigs"]) == api.contigs

    # Check dimension lengths.
    assert ds.sizes["alleles"] == 2


@pytest.mark.parametrize("aims", ["gambcolu_vs_arab", "gamb_vs_colu"])
def test_aim_calls(aims, ag3_sim_api):
    api = ag3_sim_api

    # Parametrize sample_sets.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_releases = api.releases
    parametrize_sample_sets = [
        None,
        random.choice(all_sample_sets),
        random.sample(all_sample_sets, 2),
        random.choice(all_releases),
    ]

    # Parametrize sample_query.
    parametrize_sample_query = [
        None,
        "aim_species != 'arabiensis'",
    ]

    # Run tests.
    for sample_sets, sample_query in itertools.product(
        parametrize_sample_sets, parametrize_sample_query
    ):
        # Call function to be tested.
        ds = api.aim_calls(
            aims=aims, sample_sets=sample_sets, sample_query=sample_query
        )

        # Check return type.
        assert isinstance(ds, xr.Dataset)

        # Check data variables.
        expected_data_vars = {"variant_allele", "call_genotype"}
        assert set(ds.data_vars) == expected_data_vars

        # Check coordinate variables.
        expected_coords = {"variant_contig", "variant_position", "sample_id"}
        assert set(ds.coords) == expected_coords

        # Check dimensions.
        expected_dims = {"variants", "alleles", "samples", "ploidy"}
        assert set(ds.dims) == expected_dims

        # Check variant_contig variable.
        x = ds["variant_contig"]
        assert x.dims == ("variants",)
        assert x.dtype == "uint8"

        # Check variant_position variable.
        x = ds["variant_position"]
        assert x.dims == ("variants",)
        assert (x.dtype == "int32") or (x.dtype == "int64")

        # Check variant_allele variable.
        x = ds["variant_allele"]
        assert x.dims == ("variants", "alleles")
        assert x.dtype == "S1"

        # Check call_genotype variable.
        x = ds["call_genotype"]
        assert x.dims == ("variants", "samples", "ploidy")
        assert x.dtype == "int8"

        # Check sample_id variable.
        df_samples = api.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )
        assert_array_equal(df_samples["sample_id"].values, ds["sample_id"].values)

        # Check attributes.
        assert tuple(ds.attrs["contigs"]) == api.contigs

        # Check dimension lengths.
        assert ds.sizes["samples"] == len(df_samples)
        assert ds.sizes["alleles"] == 2
        assert ds.sizes["ploidy"] == 2


def test_aim_calls_errors(ag3_sim_api):
    api = ag3_sim_api

    # Bad aims.
    with pytest.raises(ValueError):
        api.aim_calls(aims="foobar")

    # Sample query with no results.
    with pytest.raises(ValueError):
        api.aim_calls(aims="gamb_vs_colu", sample_query="country == 'Antarctica'")


@pytest.mark.parametrize("aims", ["gambcolu_vs_arab", "gamb_vs_colu"])
def test_plot_aim_heatmap(aims, ag3_sim_api):
    api = ag3_sim_api

    # Parametrize sample_sets.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    all_releases = api.releases
    parametrize_sample_sets = [
        None,
        random.choice(all_sample_sets),
        random.sample(all_sample_sets, 2),
        random.choice(all_releases),
    ]

    # Parametrize sample_query.
    parametrize_sample_query = [
        None,
        "aim_species != 'arabiensis'",
    ]

    # Run tests.
    for sample_sets, sample_query in itertools.product(
        parametrize_sample_sets, parametrize_sample_query
    ):
        # Call function to be tested.
        fig = api.plot_aim_heatmap(
            aims=aims,
            sample_sets=sample_sets,
            sample_query=sample_query,
            show=False,
        )

        # Check return type.
        assert isinstance(fig, go.Figure)
