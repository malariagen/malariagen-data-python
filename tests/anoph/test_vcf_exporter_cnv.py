import gzip
import os
import random

import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3

from malariagen_data.anoph.to_vcf_cnv import CnvVcfExporter


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return CnvVcfExporter(
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
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
        default_coverage_calls_analysis="gamb_colu",
        discordant_read_calls_analysis=ag3_sim_fixture.config[
            "DEFAULT_DISCORDANT_READ_CALLS_ANALYSIS"
        ],
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return CnvVcfExporter(
        url=af1_sim_fixture.url,
        public_url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        results_cache=af1_sim_fixture.results_cache_path.as_posix(),
        default_coverage_calls_analysis="funestus",
        discordant_read_calls_analysis=None,
    )


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


def _check_basic_vcf_structure(output_path, expected_n_variants, expected_sample_ids):
    assert os.path.exists(output_path)

    opener = gzip.open if output_path.endswith(".gz") else open
    with opener(output_path, "rt") as f:
        lines = f.readlines()

    header_lines = [line for line in lines if line.startswith("##")]
    column_line = [line for line in lines if line.startswith("#CHROM")]
    data_lines = [line for line in lines if not line.startswith("#")]

    assert header_lines[0].strip() == "##fileformat=VCFv4.3"
    assert len(column_line) == 1

    col_fields = column_line[0].strip().split("\t")
    vcf_samples = col_fields[9:]
    assert list(vcf_samples) == list(expected_sample_ids)

    assert len(data_lines) == expected_n_variants

    for line in data_lines:
        fields = line.rstrip("\n").split("\t")
        assert len(fields) == 9 + len(expected_sample_ids)
        ref, alt = fields[3], fields[4]
        assert ref == "N"
        assert alt == "<CNV>"

    return data_lines


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_hmm_to_vcf(fixture, api: CnvVcfExporter, tmp_path):
    region = random.choice(api.contigs)
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.sample(all_sample_sets, min(2, len(all_sample_sets)))

    ds = api.cnv_hmm(region=region, sample_sets=sample_sets, max_coverage_variance=None)
    n_variants = ds.sizes["variants"]
    sample_ids = ds["sample_id"].values

    output_path = str(tmp_path / "test_cnv_hmm.vcf")
    api.cnv_hmm_to_vcf(
        output_path=output_path,
        region=region,
        sample_sets=sample_sets,
        max_coverage_variance=None,
    )

    data_lines = _check_basic_vcf_structure(output_path, n_variants, sample_ids)

    # Check positions match.
    vcf_positions = sorted([int(line.split("\t")[1]) for line in data_lines])
    ds_positions = sorted(ds["variant_position"].values.tolist())
    assert vcf_positions == ds_positions

    # Check FORMAT is CN.
    for line in data_lines:
        fields = line.rstrip("\n").split("\t")
        assert fields[8] == "CN"
        assert fields[7].startswith("END=")


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_coverage_calls_to_vcf(fixture, api: CnvVcfExporter, tmp_path):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = random.choice(all_sample_sets)
    analysis = random.choice(api.coverage_calls_analysis_ids)
    region = random.choice(api.contigs)

    ds = api.cnv_coverage_calls(region=region, sample_set=sample_set, analysis=analysis)
    n_variants = ds.sizes["variants"]
    sample_ids = ds["sample_id"].values

    output_path = str(tmp_path / "test_cnv_coverage_calls.vcf")
    api.cnv_coverage_calls_to_vcf(
        output_path=output_path,
        region=region,
        sample_set=sample_set,
        analysis=analysis,
    )

    data_lines = _check_basic_vcf_structure(output_path, n_variants, sample_ids)

    ds_ids = [
        v.decode() if hasattr(v, "decode") else str(v) for v in ds["variant_id"].values
    ]
    vcf_ids = [line.split("\t")[2] for line in data_lines]
    assert sorted(vcf_ids) == sorted(ds_ids)

    for line in data_lines:
        fields = line.rstrip("\n").split("\t")
        assert fields[8] == "GT"
        assert fields[6] in ("PASS", ".")
        assert "CIPOS=" in fields[7]
        assert "CIEND=" in fields[7]
        for sample_val in fields[9:]:
            assert sample_val in ("0", "1", ".")


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_discordant_read_calls_to_vcf(fixture, api: CnvVcfExporter, tmp_path):
    contig = random.choice(api.contigs)
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.sample(all_sample_sets, min(2, len(all_sample_sets)))

    ds = api.cnv_discordant_read_calls(contigs=contig, sample_sets=sample_sets)
    n_variants = ds.sizes["variants"]
    sample_ids = ds["sample_id"].values

    output_path = str(tmp_path / "test_cnv_discordant.vcf")
    api.cnv_discordant_read_calls_to_vcf(
        output_path=output_path,
        contigs=contig,
        sample_sets=sample_sets,
    )

    data_lines = _check_basic_vcf_structure(output_path, n_variants, sample_ids)

    for line in data_lines:
        fields = line.rstrip("\n").split("\t")
        assert fields[8] == "GT"
        assert "REGION=" in fields[7]
        assert "SBM=" in fields[7]
        assert "EBM=" in fields[7]
        for sample_val in fields[9:]:
            assert sample_val in ("0", "1", ".")


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_hmm_to_vcf_overwrite(fixture, api: CnvVcfExporter, tmp_path):
    region = api.contigs[0]
    output_path = str(tmp_path / "test_cnv_hmm.vcf")

    api.cnv_hmm_to_vcf(
        output_path=output_path, region=region, max_coverage_variance=None
    )
    mtime_first = os.path.getmtime(output_path)

    # Without overwrite, should return early.
    api.cnv_hmm_to_vcf(
        output_path=output_path, region=region, max_coverage_variance=None
    )
    assert os.path.getmtime(output_path) == mtime_first

    # With overwrite, file should be rewritten.
    api.cnv_hmm_to_vcf(
        output_path=output_path,
        region=region,
        max_coverage_variance=None,
        overwrite=True,
    )
    assert os.path.exists(output_path)


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_hmm_to_vcf_gzip(fixture, api: CnvVcfExporter, tmp_path):
    region = api.contigs[0]
    output_path = str(tmp_path / "test_cnv_hmm.vcf.gz")

    api.cnv_hmm_to_vcf(
        output_path=output_path, region=region, max_coverage_variance=None
    )
    assert os.path.exists(output_path)

    with gzip.open(output_path, "rt") as f:
        first_line = f.readline()
    assert first_line.strip() == "##fileformat=VCFv4.3"
