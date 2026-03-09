import gzip
import os
import random

import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3

from malariagen_data.anoph.to_vcf import VcfExporter


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return VcfExporter(
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
    return VcfExporter(
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
def test_vcf_exporter(fixture, api: VcfExporter, tmp_path):
    region = random.choice(api.contigs)
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.sample(all_sample_sets, min(2, len(all_sample_sets)))
    site_mask = random.choice((None,) + api.site_mask_ids)

    data_params = dict(
        region=region,
        sample_sets=sample_sets,
        site_mask=site_mask,
    )

    ds = api.snp_calls(**data_params)
    n_variants = ds.sizes["variants"]
    sample_ids = ds["sample_id"].values

    output_path = str(tmp_path / "test.vcf")
    api.snp_calls_to_vcf(output_path=output_path, **data_params)

    assert os.path.exists(output_path)

    with open(output_path) as f:
        lines = f.readlines()

    header_lines = [line for line in lines if line.startswith("##")]
    column_line = [line for line in lines if line.startswith("#CHROM")]
    data_lines = [line for line in lines if not line.startswith("#")]

    # Valid VCF header.
    assert header_lines[0].strip() == "##fileformat=VCFv4.3"
    assert len(column_line) == 1

    # Sample IDs match.
    col_fields = column_line[0].strip().split("\t")
    vcf_samples = col_fields[9:]
    assert list(vcf_samples) == list(sample_ids)

    # Variant count matches.
    assert len(data_lines) == n_variants

    # Positions match.
    vcf_positions = sorted([int(line.split("\t")[1]) for line in data_lines])
    ds_positions = sorted(ds["variant_position"].values.tolist())
    assert vcf_positions == ds_positions

    # Allele values are clean strings, not byte-string representations.
    for line in data_lines:
        fields = line.split("\t")
        ref, alt = fields[3], fields[4]
        assert (
            "b'" not in ref and "b'" not in alt
        ), f"byte-string repr in REF/ALT: REF={ref!r} ALT={alt!r}"


@parametrize_with_cases("fixture,api", cases=".")
def test_vcf_exporter_overwrite(fixture, api: VcfExporter, tmp_path):
    region = api.contigs[0]
    output_path = str(tmp_path / "test.vcf")

    api.snp_calls_to_vcf(output_path=output_path, region=region)
    mtime_first = os.path.getmtime(output_path)

    # Without overwrite, should return early.
    api.snp_calls_to_vcf(output_path=output_path, region=region)
    assert os.path.getmtime(output_path) == mtime_first

    # With overwrite, file should be rewritten.
    api.snp_calls_to_vcf(output_path=output_path, region=region, overwrite=True)
    assert os.path.exists(output_path)


@parametrize_with_cases("fixture,api", cases=".")
def test_vcf_exporter_gzip(fixture, api: VcfExporter, tmp_path):
    region = api.contigs[0]
    output_path = str(tmp_path / "test.vcf.gz")

    api.snp_calls_to_vcf(output_path=output_path, region=region)
    assert os.path.exists(output_path)

    # Verify it's valid gzip.
    with gzip.open(output_path, "rt") as f:
        first_line = f.readline()
    assert first_line.strip() == "##fileformat=VCFv4.3"
