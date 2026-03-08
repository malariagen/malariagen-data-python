import random

import os
import bed_reader
import pytest
import xarray as xr
from numpy.testing import assert_array_equal
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data import adir1 as _adir1

from malariagen_data.anoph.ld import AnophelesLdAnalysis
from malariagen_data.anoph.to_plink import PlinkConverter


# The PLINK test needs a class that mixes both AnophelesLdAnalysis and
# PlinkConverter, mirroring the concrete Ag3/Af1 classes.
class _LdPlinkTestApi(AnophelesLdAnalysis, PlinkConverter):
    pass


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return _LdPlinkTestApi(
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
    return _LdPlinkTestApi(
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


@pytest.fixture
def adir1_sim_api(adir1_sim_fixture):
    return _LdPlinkTestApi(
        url=adir1_sim_fixture.url,
        public_url=adir1_sim_fixture.url,
        config_path=_adir1.CONFIG_PATH,
        major_version_number=_adir1.MAJOR_VERSION_NUMBER,
        major_version_path=_adir1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        default_site_mask="dirus",
        results_cache=adir1_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_adir1.TAXON_COLORS,
    )


# N.B., here we use pytest_cases to parametrize tests. Each
# function whose name begins with "case_" defines a set of
# inputs to the test functions. See the documentation for
# pytest_cases for more information, e.g.:
#
# https://smarie.github.io/python-pytest-cases/#basic-usage
#
# We use this approach here because we want to use fixtures
# as test parameters, which is otherwise hard to do with
# pytest alone.


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snps_ld_pruned(fixture, api: AnophelesLdAnalysis):
    """Basic LD pruning test: result is a valid dataset with fewer or equal variants."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()

    contig = random.choice(api.contigs)

    # Get unpruned dataset for comparison.
    ds_unpruned = api.biallelic_snp_calls(
        region=contig,
        sample_sets=all_sample_sets,
        min_minor_ac=2,
    )

    # Run LD pruning.
    ds_pruned = api.biallelic_snps_ld_pruned(
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

    # Second call should return identical result (cache correctness).
    ds_pruned_2 = api.biallelic_snps_ld_pruned(
        region=contig,
        sample_sets=all_sample_sets,
        min_minor_ac=2,
    )
    assert_array_equal(
        ds_pruned["variant_position"].values,
        ds_pruned_2["variant_position"].values,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snps_ld_pruned_with_n_snps(fixture, api: AnophelesLdAnalysis):
    """LD pruning with positional pre-filtering via n_snps."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    # Use first contig to avoid flaky test from random contig selection
    # hitting a contig with too few biallelic SNPs.
    contig = api.contigs[0]

    # Use a small n_snps to avoid "Not enough SNPs" with simulated data.
    n_snps = 20

    # Get the baseline variant count that biallelic_snp_calls returns for
    # these params — n_snps is a thinning hint, not a hard maximum.
    ds_base = api.biallelic_snp_calls(
        region=contig,
        sample_sets=all_sample_sets,
        n_snps=n_snps,
        min_minor_ac=1,
        site_mask=None,
        max_missing_an=None,
    )
    n_variants_base = ds_base.sizes["variants"]

    ds_pruned = api.biallelic_snps_ld_pruned(
        region=contig,
        sample_sets=all_sample_sets,
        n_snps=n_snps,
        min_minor_ac=1,
        site_mask=None,
        max_missing_an=None,
    )

    assert isinstance(ds_pruned, xr.Dataset)

    # LD pruning can only reduce the variant count, never increase it.
    assert ds_pruned.sizes["variants"] <= n_variants_base


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snps_ld_pruned_with_cohort_size(fixture, api: AnophelesLdAnalysis):
    """LD pruning with individual downsampling via max_cohort_size."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    contig = random.choice(api.contigs)

    max_cohort_size = 5

    ds_pruned = api.biallelic_snps_ld_pruned(
        region=contig,
        sample_sets=all_sample_sets,
        max_cohort_size=max_cohort_size,
        min_minor_ac=1,
    )

    assert isinstance(ds_pruned, xr.Dataset)

    # Result should have <= max_cohort_size samples.
    assert ds_pruned.sizes["samples"] <= max_cohort_size


@parametrize_with_cases("fixture,api", cases=".")
def test_biallelic_snps_ld_pruned_to_plink(fixture, api: _LdPlinkTestApi, tmp_path):
    """Test that LD-pruned data can be exported to PLINK binary format."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    contig = random.choice(api.contigs)

    # Export LD-pruned data to PLINK.
    plink_path = api.biallelic_snps_ld_pruned_to_plink(
        output_dir=str(tmp_path),
        region=contig,
        sample_sets=all_sample_sets,
        min_minor_ac=1,
    )

    # Check output files exist (.bed, .bim, .fam).
    assert os.path.exists(f"{plink_path}.bed")
    assert os.path.exists(f"{plink_path}.bim")
    assert os.path.exists(f"{plink_path}.fam")

    # Read the PLINK files and validate against the LD-pruned dataset.
    bed = bed_reader.open_bed(f"{plink_path}.bed")

    ds_pruned = api.biallelic_snps_ld_pruned(
        region=contig,
        sample_sets=all_sample_sets,
        min_minor_ac=1,
    )

    # Variant count should match exactly (no sites are dropped during
    # PLINK export because biallelic_snp_calls already guarantees
    # both alleles are observed, i.e. variation exists).
    assert bed.shape[1] == ds_pruned.sizes["variants"]

    # Sample count should match.
    assert bed.shape[0] == ds_pruned.sizes["samples"]

    # Sample IDs should match.
    assert_array_equal(bed.iid, ds_pruned["sample_id"].values)

    # PLINK positions should match pruned positions exactly.
    assert set(bed.bp_position) == set(ds_pruned["variant_position"].values)

    # Chromosome IDs should match (coerce to str to match types).
    assert set(bed.chromosome) == set(ds_pruned["variant_contig"].values.astype(str))

    # Major and minor alleles should match.
    assert set(bed.allele_1) == set(
        ds_pruned["variant_allele"].values[:, 0].astype(str)
    )
    assert set(bed.allele_2) == set(
        ds_pruned["variant_allele"].values[:, 1].astype(str)
    )

    # Second call with overwrite=False should return the same path (no recompute).
    plink_path_2 = api.biallelic_snps_ld_pruned_to_plink(
        output_dir=str(tmp_path),
        region=contig,
        sample_sets=all_sample_sets,
        min_minor_ac=1,
    )
    assert plink_path_2 == plink_path
