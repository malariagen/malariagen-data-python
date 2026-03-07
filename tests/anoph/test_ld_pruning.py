import random
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3

from malariagen_data.anoph.ld_pruning import AnophelesLdPruning

import numpy as np
import xarray as xr
from numpy.testing import assert_array_equal


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesLdPruning(
        url=ag3_sim_fixture.url,
        public_url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
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
        gff_gene_type="gene",
        gff_gene_name_attribute="Name",
        gff_default_attributes=("ID", "Parent", "Name", "description"),
        default_site_mask="gamb_colu_arab",
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_ag3.TAXON_COLORS,
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesLdPruning(
        url=af1_sim_fixture.url,
        public_url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        default_site_mask="funestus",
        results_cache=af1_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_af1.TAXON_COLORS,
    )


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


@parametrize_with_cases("fixture,api", cases=".")
def test_ld_prune_basic(fixture, api: AnophelesLdPruning):
    """Test that ld_prune returns a valid xr.Dataset with fewer variants."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    contig = random.choice(api.contigs)

    # Get unpruned dataset for comparison.
    ds_unpruned = api.biallelic_snp_calls(
        region=contig,
        sample_sets=all_sample_sets,
        min_minor_ac=2,
    )

    # Run LD pruning.
    ds_pruned = api.ld_prune(
        region=contig,
        sample_sets=all_sample_sets,
        min_minor_ac=2,
    )

    # Result should be an xarray Dataset.
    assert isinstance(ds_pruned, xr.Dataset)

    # Pruned variants should be <= unpruned variants.
    assert ds_pruned.sizes["variants"] <= ds_unpruned.sizes["variants"]

    # Sample IDs should be unchanged.
    assert_array_equal(
        ds_pruned["sample_id"].values,
        ds_unpruned["sample_id"].values,
    )

    # All retained positions should be a subset of the original positions.
    pruned_positions = set(ds_pruned["variant_position"].values)
    unpruned_positions = set(ds_unpruned["variant_position"].values)
    assert pruned_positions.issubset(unpruned_positions)


@parametrize_with_cases("fixture,api", cases=".")
def test_ld_prune_caching(fixture, api: AnophelesLdPruning):
    """Test that caching works (second call returns identical result)."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    contig = random.choice(api.contigs)

    # First call.
    ds_pruned_1 = api.ld_prune(
        region=contig,
        sample_sets=all_sample_sets,
        min_minor_ac=2,
    )

    # Second call should return identical result (from cache).
    ds_pruned_2 = api.ld_prune(
        region=contig,
        sample_sets=all_sample_sets,
        min_minor_ac=2,
    )

    assert_array_equal(
        ds_pruned_1["variant_position"].values,
        ds_pruned_2["variant_position"].values,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_ld_prune_custom_params(fixture, api: AnophelesLdPruning):
    """Test ld_prune with custom r2 threshold and window parameters."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    contig = random.choice(api.contigs)

    # Stricter threshold should remove more SNPs.
    ds_strict = api.ld_prune(
        region=contig,
        sample_sets=all_sample_sets[:1],
        r2_threshold=0.05,
        window_size=200,
        window_step=100,
        random_seed=42,
    )

    ds_relaxed = api.ld_prune(
        region=contig,
        sample_sets=all_sample_sets[:1],
        r2_threshold=0.5,
        window_size=200,
        window_step=100,
        random_seed=42,
    )

    # Stricter threshold should keep fewer or equal SNPs.
    assert ds_strict.sizes["variants"] <= ds_relaxed.sizes["variants"]


@parametrize_with_cases("fixture,api", cases=".")
def test_ld_prune_dataset_structure(fixture, api: AnophelesLdPruning):
    """Test that the pruned dataset has the correct structure."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    contig = random.choice(api.contigs)

    ds_pruned = api.ld_prune(
        region=contig,
        sample_sets=all_sample_sets,
        min_minor_ac=1,
    )

    # Check expected coordinates.
    assert "sample_id" in ds_pruned.coords
    assert "variant_position" in ds_pruned.coords
    assert "variant_contig" in ds_pruned.coords

    # Check expected data variables.
    assert "variant_allele" in ds_pruned.data_vars
    assert "call_genotype" in ds_pruned.data_vars

    # Check dimensions.
    assert "variants" in ds_pruned.dims
    assert "samples" in ds_pruned.dims
    assert "ploidy" in ds_pruned.dims
    assert "alleles" in ds_pruned.dims

    # Check alleles are biallelic.
    assert ds_pruned.sizes["alleles"] == 2


@parametrize_with_cases("fixture,api", cases=".")
def test_ld_prune_plink_compatibility(fixture, api: AnophelesLdPruning):
    """Test that the pruned dataset has all variables required by PlinkConverter."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    contig = random.choice(api.contigs)

    ds_pruned = api.ld_prune(
        region=contig,
        sample_sets=all_sample_sets,
        min_minor_ac=1,
    )

    # Verify the pruned dataset has all variables required by PlinkConverter.
    assert "call_genotype" in ds_pruned
    assert "variant_allele" in ds_pruned
    assert "variant_contig" in ds_pruned.coords
    assert "variant_position" in ds_pruned.coords
    assert "sample_id" in ds_pruned.coords

    # Verify shapes are internally consistent.
    n_variants = ds_pruned.sizes["variants"]
    n_samples = ds_pruned.sizes["samples"]
    assert ds_pruned["call_genotype"].shape == (n_variants, n_samples, 2)
    assert ds_pruned["variant_allele"].shape == (n_variants, 2)
