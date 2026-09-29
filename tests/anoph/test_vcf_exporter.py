import gzip
import os
import random

import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3

from malariagen_data.anoph.to_vcf import SnpVcfExporter


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return SnpVcfExporter(
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
    return SnpVcfExporter(
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


def _pick_single_sample_query(api: SnpVcfExporter, sample_sets):
    """Pick one sample from the given sample sets and build a
    `sample_query` that scopes a VCF export down to just that sample,
    which is the only supported usage of `snp_calls_to_vcf()` (it is
    used to generate a single-sample VCF for IGV)."""
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    sample_id = str(df_samples["sample_id"].iloc[0])
    sample_query = f"sample_id == '{sample_id}'"
    return sample_id, sample_query


@parametrize_with_cases("fixture,api", cases=".")
def test_vcf_exporter(fixture, api: SnpVcfExporter, tmp_path):
    region = random.choice(api.contigs)
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.sample(all_sample_sets, min(2, len(all_sample_sets)))
    site_mask = random.choice((None,) + api.site_mask_ids)
    sample_id, sample_query = _pick_single_sample_query(api, sample_sets)

    data_params = dict(
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        site_mask=site_mask,
    )

    ds = api.snp_calls(**data_params)
    n_variants = ds.sizes["variants"]
    sample_ids = ds["sample_id"].values

    # snp_calls_to_vcf() only supports exporting a single sample at a
    # time (its only current use case is generating a per-sample VCF
    # for IGV), so sample_query must narrow the dataset down to one
    # sample.
    assert list(sample_ids) == [sample_id]

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
def test_vcf_exporter_requires_sample_sets_and_query(
    fixture, api: SnpVcfExporter, tmp_path
):
    # sample_sets and sample_query are required: a VCF without an
    # explicit cohort/sample basis for the calls it contains is not
    # meaningful.
    region = api.contigs[0]
    output_path = str(tmp_path / "test_missing_args.vcf")

    with pytest.raises(TypeError):
        api.snp_calls_to_vcf(output_path=output_path, region=region)  # type: ignore[call-arg]

    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = [all_sample_sets[0]]
    with pytest.raises(TypeError):
        api.snp_calls_to_vcf(  # type: ignore[call-arg]
            output_path=output_path, region=region, sample_sets=sample_sets
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_vcf_exporter_empty_sample_sets(fixture, api: SnpVcfExporter, tmp_path):
    region = api.contigs[0]
    output_path = str(tmp_path / "test_empty_sample_sets.vcf")

    with pytest.raises(ValueError, match="sample_sets must be provided"):
        api.snp_calls_to_vcf(
            output_path=output_path,
            region=region,
            sample_sets=[],
            sample_query="sample_id == 'anything'",
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_vcf_exporter_empty_sample_query(fixture, api: SnpVcfExporter, tmp_path):
    region = api.contigs[0]
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = [all_sample_sets[0]]
    output_path = str(tmp_path / "test_empty_sample_query.vcf")

    with pytest.raises(ValueError, match="sample_query must be provided"):
        api.snp_calls_to_vcf(
            output_path=output_path,
            region=region,
            sample_sets=sample_sets,
            sample_query="",
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_vcf_exporter_multi_sample_query_rejected(
    fixture, api: SnpVcfExporter, tmp_path
):
    # sample_query must select a single sample via equality on
    # sample_id; a query that could match more than one sample (even a
    # syntactically simple one like a country filter) is rejected
    # up front, before touching any data.
    region = api.contigs[0]
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = [all_sample_sets[0]]
    output_path = str(tmp_path / "test_multi_sample_query.vcf")

    for bad_query in [
        "country == 'Uganda'",
        "sample_id in ['AB0085-C', 'AB0086-C']",
        "sample_id != 'AB0085-C'",
    ]:
        with pytest.raises(ValueError, match="does not select a single sample"):
            api.snp_calls_to_vcf(
                output_path=output_path,
                region=region,
                sample_sets=sample_sets,
                sample_query=bad_query,
            )


@parametrize_with_cases("fixture,api", cases=".")
def test_vcf_exporter_sample_query_matches_zero_samples(
    fixture, api: SnpVcfExporter, tmp_path
):
    # A query in the right shape but for a sample_id that isn't part of
    # the given sample_sets should still be rejected, not silently
    # produce a zero-sample VCF. In practice snp_calls() itself already
    # raises for a query matching no samples, before our own post-hoc
    # sample-count check ever runs; either way this must not succeed.
    region = api.contigs[0]
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = [all_sample_sets[0]]
    output_path = str(tmp_path / "test_zero_sample_query.vcf")

    with pytest.raises(ValueError):
        api.snp_calls_to_vcf(
            output_path=output_path,
            region=region,
            sample_sets=sample_sets,
            sample_query="sample_id == 'not_a_real_sample_id'",
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_vcf_exporter_overwrite(fixture, api: SnpVcfExporter, tmp_path):
    region = api.contigs[0]
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = [all_sample_sets[0]]
    _, sample_query = _pick_single_sample_query(api, sample_sets)
    output_path = str(tmp_path / "test.vcf")

    api.snp_calls_to_vcf(
        output_path=output_path,
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
    )
    mtime_first = os.path.getmtime(output_path)

    # Without overwrite, should return early.
    api.snp_calls_to_vcf(
        output_path=output_path,
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
    )
    assert os.path.getmtime(output_path) == mtime_first

    # With overwrite, file should be rewritten.
    api.snp_calls_to_vcf(
        output_path=output_path,
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        overwrite=True,
    )
    assert os.path.exists(output_path)


@parametrize_with_cases("fixture,api", cases=".")
def test_vcf_exporter_gzip(fixture, api: SnpVcfExporter, tmp_path):
    region = api.contigs[0]
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = [all_sample_sets[0]]
    _, sample_query = _pick_single_sample_query(api, sample_sets)
    output_path = str(tmp_path / "test.vcf.gz")

    api.snp_calls_to_vcf(
        output_path=output_path,
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
    )
    assert os.path.exists(output_path)

    # Verify it's valid gzip.
    with gzip.open(output_path, "rt") as f:
        first_line = f.readline()
    assert first_line.strip() == "##fileformat=VCFv4.3"


@parametrize_with_cases("fixture,api", cases=".")
def test_vcf_exporter_fields(fixture, api: SnpVcfExporter, tmp_path):
    region = api.contigs[0]
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = [all_sample_sets[0]]
    _, sample_query = _pick_single_sample_query(api, sample_sets)

    # Test with additional FORMAT fields.
    output_path = str(tmp_path / "test_fields.vcf")
    api.snp_calls_to_vcf(
        output_path=output_path,
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        fields=("GT", "GQ"),
    )

    with open(output_path) as f:
        lines = f.readlines()

    # Check FORMAT header lines.
    format_lines = [line for line in lines if line.startswith("##FORMAT")]
    assert len(format_lines) == 2
    assert any("ID=GT" in line for line in format_lines)
    assert any("ID=GQ" in line for line in format_lines)

    # Check FORMAT column value.
    data_lines = [line for line in lines if not line.startswith("#")]
    assert len(data_lines) > 0
    first_data = data_lines[0].strip().split("\t")
    assert first_data[8] == "GT:GQ"

    # Each sample field should have two colon-separated values.
    for sample_val in first_data[9:]:
        parts = sample_val.split(":")
        assert len(parts) == 2, f"Expected GT:GQ, got {sample_val!r}"


@parametrize_with_cases("fixture,api", cases=".")
def test_vcf_exporter_fields_as_set(fixture, api: SnpVcfExporter, tmp_path):
    # A `set` has no guaranteed iteration order. The FORMAT column must
    # still come out in the same fixed order (GT, GQ, AD, MQ) that the
    # per-sample values are actually written in, regardless of what
    # order `fields` iterates in as passed by the caller. Previously
    # the FORMAT column used the caller's (here, arbitrary) order while
    # values were always written GT:GQ:AD:MQ, so a `set` could easily
    # produce a VCF whose header lied about which value was which —
    # e.g. declaring the first value as "MQ" when it was actually the
    # GT string, which bcftools then fails to parse.
    region = api.contigs[0]
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = [all_sample_sets[0]]
    _, sample_query = _pick_single_sample_query(api, sample_sets)

    output_path = str(tmp_path / "test_fields_set.vcf")
    api.snp_calls_to_vcf(
        output_path=output_path,
        region=region,
        sample_sets=sample_sets,
        sample_query=sample_query,
        fields={"GT", "GQ", "AD", "MQ"},
    )

    with open(output_path) as f:
        lines = f.readlines()

    data_lines = [line for line in lines if not line.startswith("#")]
    assert len(data_lines) > 0

    for line in data_lines:
        fields_row = line.strip().split("\t")
        # FORMAT column must always be in this fixed order.
        assert fields_row[8] == "GT:GQ:AD:MQ"

        for sample_val in fields_row[9:]:
            gt, gq, ad, mq = sample_val.split(":")
            # GT is the only field allowed to contain "/"; if it ended
            # up anywhere else (or something else ended up in its
            # place), that's exactly the bug this test guards against.
            assert gt == "./." or "/" in gt
            assert "/" not in gq
            assert "/" not in ad
            assert "/" not in mq


@parametrize_with_cases("fixture,api", cases=".")
def test_vcf_exporter_fields_gt_required(fixture, api: SnpVcfExporter, tmp_path):
    region = api.contigs[0]
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = [all_sample_sets[0]]
    _, sample_query = _pick_single_sample_query(api, sample_sets)
    output_path = str(tmp_path / "test_no_gt.vcf")

    with pytest.raises(ValueError, match="GT must be included"):
        api.snp_calls_to_vcf(
            output_path=output_path,
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            fields=("GQ",),
        )
