import random
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3

from malariagen_data.anoph.admixture import AnophelesAdmixture

import os
import bed_reader


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesAdmixture(
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
    return AnophelesAdmixture(
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
def test_prepare_admixture_basic(fixture, api: AnophelesAdmixture, tmp_path):
    """Test that prepare_admixture produces valid PLINK output."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()

    results = api.prepare_admixture(
        output_dir=str(tmp_path),
        region=random.choice(api.contigs),
        sample_sets=random.sample(all_sample_sets, 2),
        site_mask=random.choice((None,) + api.site_mask_ids),
        min_minor_ac=1,
        max_missing_an=1,
        random_seed=42,
    )

    # Check result keys.
    assert "plink_path" in results
    assert "n_samples" in results
    assert "n_snps_after_ld" in results
    assert "n_snps_final" in results
    assert results["from_cache"] is False

    # Check that PLINK files were created.
    plink_path = results["plink_path"]
    assert os.path.exists(f"{plink_path}.bed")
    assert os.path.exists(f"{plink_path}.bim")
    assert os.path.exists(f"{plink_path}.fam")

    # Check that LD pruning reduced SNPs.
    assert results["n_snps_final"] <= results["n_snps_after_ld"]
    assert results["n_samples"] > 0

    # Validate PLINK file contents.
    bed = bed_reader.open_bed(f"{plink_path}.bed")
    assert bed.shape[0] == results["n_samples"]
    assert bed.shape[1] == results["n_snps_final"]


@parametrize_with_cases("fixture,api", cases=".")
def test_prepare_admixture_max_snps(fixture, api: AnophelesAdmixture, tmp_path):
    """Test that max_snps downsampling works correctly."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()

    results = api.prepare_admixture(
        output_dir=str(tmp_path),
        region=random.choice(api.contigs),
        sample_sets=all_sample_sets[:1],
        max_snps=10,
        random_seed=42,
        overwrite=True,
    )

    # Final SNPs should be at most max_snps.
    assert results["n_snps_final"] <= 10


@parametrize_with_cases("fixture,api", cases=".")
def test_prepare_admixture_cache(fixture, api: AnophelesAdmixture, tmp_path):
    """Test that file-based caching works (second call uses cached result)."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    contig = random.choice(api.contigs)

    params = dict(
        output_dir=str(tmp_path),
        region=contig,
        sample_sets=all_sample_sets[:1],
        random_seed=42,
    )

    # First call creates files.
    results1 = api.prepare_admixture(**params)
    assert results1["from_cache"] is False

    # Second call returns cached result.
    results2 = api.prepare_admixture(**params)
    assert results2["from_cache"] is True
