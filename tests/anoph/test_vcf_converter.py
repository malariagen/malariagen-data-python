import random
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data import adir1 as _adir1

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


@pytest.fixture
def adir1_sim_api(adir1_sim_fixture):
    return VcfConverter(
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
def test_vcf_converter(fixture, api: VcfConverter, tmp_path):
    """Test that snp_calls_to_vcf() produces a valid VCF file."""
    # Parameters for selecting input data.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    region = random.choice(api.contigs)
    site_mask = random.choice((None,) + api.site_mask_ids)

    # Export to VCF.
    vcf_path = api.snp_calls_to_vcf(
        output_dir=str(tmp_path),
        region=region,
        sample_sets=all_sample_sets,
        site_mask=site_mask,
    )

    # Check VCF file exists.
    assert os.path.exists(vcf_path)
    assert vcf_path.endswith(".vcf")

    # Read the VCF file content.
    with open(vcf_path) as f:
        lines = f.readlines()

    # Check header starts with the fileformat line.
    assert lines[0].strip() == "##fileformat=VCFv4.3"

    # Find the column header line.
    header_line = None
    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("#CHROM"):
            header_line = line.strip()
            header_idx = idx
            break
    assert header_line is not None, "No #CHROM header line found"
    assert header_idx is not None

    # Check column header has the expected fixed columns.
    header_fields = header_line.split("\t")
    expected_fixed = [
        "#CHROM",
        "POS",
        "ID",
        "REF",
        "ALT",
        "QUAL",
        "FILTER",
        "INFO",
        "FORMAT",
    ]
    assert header_fields[:9] == expected_fixed

    # Load the corresponding SNP data to check against.
    ds = api.snp_calls(
        region=region,
        sample_sets=all_sample_sets,
        site_mask=site_mask,
    )

    sample_ids = ds["sample_id"].values
    if sample_ids.dtype.kind == "S":
        sample_ids = sample_ids.astype("U")

    # Check sample IDs in VCF header match dataset.
    vcf_sample_ids = header_fields[9:]
    assert vcf_sample_ids == list(sample_ids)

    # Count data lines (non-header, non-comment).
    data_lines = [ln for ln in lines[header_idx + 1 :] if not ln.startswith("#")]
    n_variants = ds.sizes["variants"]
    assert len(data_lines) == n_variants

    # Check a few variant positions match.
    if n_variants > 0:
        positions = ds["variant_position"].values
        for i, data_line in enumerate(data_lines[:5]):
            fields = data_line.strip().split("\t")
            vcf_pos = int(fields[1])
            assert vcf_pos == positions[i]

            # Check REF is a proper string, not a byte artifact.
            ref = fields[3]
            assert not ref.startswith("b'"), f"Byte artifact in REF: {ref}"
            assert ref in ("A", "C", "G", "T", "N", "")


@parametrize_with_cases("fixture,api", cases=".")
def test_vcf_converter_overwrite(fixture, api: VcfConverter, tmp_path):
    """Test that overwrite flag works correctly."""
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    region = random.choice(api.contigs)

    # First export.
    vcf_path = api.snp_calls_to_vcf(
        output_dir=str(tmp_path),
        region=region,
        sample_sets=all_sample_sets,
    )
    assert os.path.exists(vcf_path)
    mtime1 = os.path.getmtime(vcf_path)

    # Second call without overwrite should return existing file.
    vcf_path2 = api.snp_calls_to_vcf(
        output_dir=str(tmp_path),
        region=region,
        sample_sets=all_sample_sets,
        overwrite=False,
    )
    assert vcf_path2 == vcf_path
    mtime2 = os.path.getmtime(vcf_path)
    assert mtime2 == mtime1  # File should not be modified.

    # Third call with overwrite should regenerate.
    vcf_path3 = api.snp_calls_to_vcf(
        output_dir=str(tmp_path),
        region=region,
        sample_sets=all_sample_sets,
        overwrite=True,
    )
    assert vcf_path3 == vcf_path
    assert os.path.exists(vcf_path3)
