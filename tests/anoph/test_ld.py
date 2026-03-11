import random

import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3

from malariagen_data.anoph.ld import AnophelesLdAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesLdAnalysis(
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
    return AnophelesLdAnalysis(
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
def test_ld_pruning_returns_fewer_snps(fixture, api: AnophelesLdAnalysis):
    region = random.choice(api.contigs)
    site_mask = random.choice(api.site_mask_ids)
    ds_full = api.biallelic_snp_calls(
        region=region,
        site_mask=site_mask,
        min_minor_ac=1,
        max_missing_an=0,
    )
    n_available = ds_full.sizes["variants"]
    if n_available < 10:
        pytest.skip("Not enough variants for LD pruning test")

    n_snps = min(n_available, 200)

    ds_pruned = api.biallelic_snp_calls_ld_pruned(
        region=region,
        n_snps=n_snps,
        site_mask=site_mask,
        min_minor_ac=1,
        max_missing_an=0,
    )

    # Pruned dataset should have fewer or equal variants.
    assert ds_pruned.sizes["variants"] <= n_snps
    assert ds_pruned.sizes["variants"] > 0


@parametrize_with_cases("fixture,api", cases=".")
def test_ld_pruned_dataset_structure(fixture, api: AnophelesLdAnalysis):
    region = random.choice(api.contigs)
    site_mask = random.choice(api.site_mask_ids)
    ds_full = api.biallelic_snp_calls(
        region=region,
        site_mask=site_mask,
        min_minor_ac=1,
        max_missing_an=0,
    )
    n_available = ds_full.sizes["variants"]
    if n_available < 10:
        pytest.skip("Not enough variants for LD pruning test")

    n_snps = min(n_available, 200)

    ds_pruned = api.biallelic_snp_calls_ld_pruned(
        region=region,
        n_snps=n_snps,
        site_mask=site_mask,
        min_minor_ac=1,
        max_missing_an=0,
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
def test_ld_pruned_plink_compatibility(fixture, api: AnophelesLdAnalysis):
    region = random.choice(api.contigs)
    site_mask = random.choice(api.site_mask_ids)
    ds_full = api.biallelic_snp_calls(
        region=region,
        site_mask=site_mask,
        min_minor_ac=1,
        max_missing_an=0,
    )
    n_available = ds_full.sizes["variants"]
    if n_available < 10:
        pytest.skip("Not enough variants for LD pruning test")

    n_snps = min(n_available, 200)

    ds_pruned = api.biallelic_snp_calls_ld_pruned(
        region=region,
        n_snps=n_snps,
        site_mask=site_mask,
        min_minor_ac=1,
        max_missing_an=0,
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


@parametrize_with_cases("fixture,api", cases=".")
def test_ld_pruning_threshold_sensitivity(fixture, api: AnophelesLdAnalysis):
    region = random.choice(api.contigs)
    site_mask = random.choice(api.site_mask_ids)
    ds_full = api.biallelic_snp_calls(
        region=region,
        site_mask=site_mask,
        min_minor_ac=1,
        max_missing_an=0,
    )
    n_available = ds_full.sizes["variants"]
    if n_available < 10:
        pytest.skip("Not enough variants for LD pruning test")

    n_snps = min(n_available, 200)

    try:
        ds_strict = api.biallelic_snp_calls_ld_pruned(
            region=region,
            n_snps=n_snps,
            site_mask=site_mask,
            min_minor_ac=1,
            max_missing_an=0,
            ld_threshold=0.1,
        )
    except ValueError:
        pytest.skip("Too few variants survive strict LD pruning")

    ds_lenient = api.biallelic_snp_calls_ld_pruned(
        region=region,
        n_snps=n_snps,
        site_mask=site_mask,
        min_minor_ac=1,
        max_missing_an=0,
        ld_threshold=0.5,
    )

    # A stricter threshold should retain fewer or equal variants.
    assert ds_strict.sizes["variants"] <= ds_lenient.sizes["variants"]
