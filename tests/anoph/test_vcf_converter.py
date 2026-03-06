import random

import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3

from malariagen_data.anoph.to_vcf import VcfConverter

import os


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return VcfConverter(
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
    return VcfConverter(
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
def test_vcf_converter(fixture, api: VcfConverter, tmp_path):
    # Parameters for selecting input data, filtering, and converting.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()

    data_params = dict(
        region=random.choice(api.contigs),
        sample_sets=random.sample(all_sample_sets, 2),
        site_mask=random.choice((None,) + api.site_mask_ids),
        min_minor_ac=1,
        max_missing_an=1,
        thin_offset=1,
        random_seed=random.randint(1, 2000),
    )

    # Load a ds containing the randomly generated samples and regions
    # to get the number of available snps to subset from.
    ds = api.biallelic_snp_calls(
        **data_params,
    )

    n_snps_available = ds.sizes["variants"]
    n_snps = random.randint(1, n_snps_available)

    # Define vcf params.
    vcf_params = dict(output_dir=str(tmp_path), n_snps=n_snps, **data_params)

    # Make the vcf file.
    vcf_file_path = api.biallelic_snps_to_vcf(**vcf_params)

    # Test to see if the .vcf output file exists.
    assert os.path.exists(vcf_file_path)
    assert vcf_file_path.endswith(".vcf")

    # Read the VCF file and validate its structure.
    with open(vcf_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # VCF files must start with the fileformat header.
    assert lines[0].startswith("##fileformat=VCF")

    # Find the header line (starts with #CHROM).
    header_line = None
    header_line_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("#CHROM"):
            header_line = line.strip()
            header_line_idx = idx
            break

    assert header_line is not None, "VCF file missing #CHROM header line"

    # Validate mandatory VCF columns are present.
    header_fields = header_line.split("\t")
    assert header_fields[0] == "#CHROM"
    assert header_fields[1] == "POS"
    assert header_fields[2] == "ID"
    assert header_fields[3] == "REF"
    assert header_fields[4] == "ALT"

    # Load a ds containing the same data exported to VCF to test against.
    ds_test = api.biallelic_snp_calls(
        **data_params,
        n_snps=n_snps,
    )

    # Validate that sample IDs in the VCF header match the dataset.
    # VCF sample columns start after FORMAT (index 9).
    if len(header_fields) > 9:
        vcf_samples = header_fields[9:]
        expected_samples = ds_test.sample_id.values.tolist()
        assert len(vcf_samples) > 0
        # Check that all expected samples appear in the VCF header.
        for sample in vcf_samples:
            assert sample in expected_samples

    # Validate that data lines exist after the header.
    data_lines = lines[header_line_idx + 1 :]
    assert len(data_lines) > 0, "VCF file contains no data records"

    # Test that the overwrite=False caching mechanism works.
    vcf_file_path_cached = api.biallelic_snps_to_vcf(**vcf_params)
    assert vcf_file_path_cached == vcf_file_path
