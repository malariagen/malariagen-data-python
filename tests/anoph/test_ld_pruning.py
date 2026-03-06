import random
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3

from malariagen_data.anoph.ld_pruning import AnophelesLdPruning

import numpy as np


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
    """Test that ld_prune returns valid results."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()

    results = api.ld_prune(
        region=random.choice(api.contigs),
        sample_sets=random.sample(all_sample_sets, 2),
        site_mask=random.choice((None,) + api.site_mask_ids),
        min_minor_ac=1,
        max_missing_an=1,
        random_seed=42,
    )

    # Check result keys.
    assert "gn" in results
    assert "samples" in results
    assert "variant_position" in results
    assert "variant_contig" in results
    assert "n_snps_before" in results
    assert "n_snps_after" in results

    # Check that pruning reduced the number of SNPs.
    assert results["n_snps_after"] <= results["n_snps_before"]

    # Check shapes are consistent.
    assert results["gn"].shape[0] == results["n_snps_after"]
    assert results["gn"].shape[0] == len(results["variant_position"])
    assert results["gn"].shape[0] == len(results["variant_contig"])
    assert results["gn"].shape[1] == len(results["samples"])

    # Check that genotype values are valid (-1, 0, 1, 2).
    assert np.all(np.isin(results["gn"], [-1, 0, 1, 2]))


@parametrize_with_cases("fixture,api", cases=".")
def test_ld_prune_custom_params(fixture, api: AnophelesLdPruning):
    """Test ld_prune with custom r2 threshold and window parameters."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    contig = random.choice(api.contigs)

    # Stricter threshold should remove more SNPs.
    results_strict = api.ld_prune(
        region=contig,
        sample_sets=all_sample_sets[:1],
        r2_threshold=0.05,
        window_size=200,
        window_step=100,
        random_seed=42,
    )

    results_relaxed = api.ld_prune(
        region=contig,
        sample_sets=all_sample_sets[:1],
        r2_threshold=0.5,
        window_size=200,
        window_step=100,
        random_seed=42,
    )

    # Stricter threshold should keep fewer or equal SNPs.
    assert results_strict["n_snps_after"] <= results_relaxed["n_snps_after"]
