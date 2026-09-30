import gzip
import os
import random

import dask.array as da
import numpy as np
import pytest
import xarray as xr
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3

from malariagen_data.anoph.to_coverage import CoverageExporter


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return CoverageExporter(
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
    return CoverageExporter(
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


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_hmm_coverage_to_bedgraph(fixture, api: CoverageExporter, tmp_path):
    region = random.choice(api.contigs)
    sample = str(np.random.choice(api.sample_metadata()["sample_id"]))

    output_path = str(tmp_path / "test_coverage.bedgraph")
    api.cnv_hmm_coverage_to_bedgraph(
        output_path=output_path,
        region=region,
        sample=sample,
    )

    assert os.path.exists(output_path)

    with open(output_path) as f:
        lines = f.readlines()

    assert lines[0].startswith("track type=bedGraph")

    data_lines = lines[1:]
    assert len(data_lines) > 0

    for line in data_lines:
        fields = line.rstrip("\n").split("\t")
        assert len(fields) == 4
        chrom, start, end, value = fields
        assert chrom == region
        assert int(start) >= 0
        assert int(end) > int(start)
        # Should be a valid number and non-negative (masked/missing
        # values are skipped rather than written).
        assert float(value) >= 0


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_hmm_coverage_to_bedgraph_raw_cov(fixture, api: CoverageExporter, tmp_path):
    region = api.contigs[0]
    sample = str(api.sample_metadata()["sample_id"].iloc[0])

    output_path = str(tmp_path / "test_coverage_raw.bedgraph")
    api.cnv_hmm_coverage_to_bedgraph(
        output_path=output_path,
        region=region,
        sample=sample,
        field="RawCov",
    )
    assert os.path.exists(output_path)


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_hmm_coverage_to_bedgraph_invalid_field(
    fixture, api: CoverageExporter, tmp_path
):
    region = api.contigs[0]
    sample = str(api.sample_metadata()["sample_id"].iloc[0])

    output_path = str(tmp_path / "test_coverage_bad.bedgraph")
    with pytest.raises(ValueError, match="Unknown coverage field"):
        api.cnv_hmm_coverage_to_bedgraph(
            output_path=output_path,
            region=region,
            sample=sample,
            field="NotAField",
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_hmm_coverage_to_bedgraph_unknown_sample(
    fixture, api: CoverageExporter, tmp_path
):
    region = api.contigs[0]
    output_path = str(tmp_path / "test_coverage_unknown.bedgraph")
    with pytest.raises(ValueError, match="No data found for sample"):
        api.cnv_hmm_coverage_to_bedgraph(
            output_path=output_path,
            region=region,
            sample="not_a_real_sample",
        )


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_hmm_coverage_to_bedgraph_overwrite(
    fixture, api: CoverageExporter, tmp_path
):
    region = api.contigs[0]
    sample = str(api.sample_metadata()["sample_id"].iloc[0])
    output_path = str(tmp_path / "test_coverage.bedgraph")

    api.cnv_hmm_coverage_to_bedgraph(
        output_path=output_path, region=region, sample=sample
    )
    mtime_first = os.path.getmtime(output_path)

    api.cnv_hmm_coverage_to_bedgraph(
        output_path=output_path, region=region, sample=sample
    )
    assert os.path.getmtime(output_path) == mtime_first

    api.cnv_hmm_coverage_to_bedgraph(
        output_path=output_path, region=region, sample=sample, overwrite=True
    )
    assert os.path.exists(output_path)


@parametrize_with_cases("fixture,api", cases=".")
def test_cnv_hmm_coverage_to_bedgraph_gzip(fixture, api: CoverageExporter, tmp_path):
    region = api.contigs[0]
    sample = str(api.sample_metadata()["sample_id"].iloc[0])
    output_path = str(tmp_path / "test_coverage.bedgraph.gz")

    api.cnv_hmm_coverage_to_bedgraph(
        output_path=output_path, region=region, sample=sample
    )
    assert os.path.exists(output_path)

    with gzip.open(output_path, "rt") as f:
        first_line = f.readline()
    assert first_line.startswith("track type=bedGraph")


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_coverage_to_bedgraph(fixture, api: CoverageExporter, tmp_path):
    region = random.choice(api.contigs)
    sample = str(np.random.choice(api.sample_metadata()["sample_id"]))

    output_path = str(tmp_path / "test_snp_coverage.bedgraph")
    api.snp_calls_coverage_to_bedgraph(
        output_path=output_path,
        region=region,
        sample=sample,
    )

    assert os.path.exists(output_path)

    with open(output_path) as f:
        lines = f.readlines()

    assert lines[0].startswith("track type=bedGraph")

    # N.B., the simulated AD data is filled with a missing-value
    # sentinel (-1) for every site, so no data lines are expected here;
    # this exercises that missing values are skipped rather than
    # written as misleading zero-depth entries. The arithmetic for
    # sites that do have depth is checked separately below, using a
    # synthetic dataset.
    for line in lines[1:]:
        fields = line.rstrip("\n").split("\t")
        assert len(fields) == 4
        chrom, start, end, value = fields
        assert chrom == region
        assert int(end) == int(start) + 1
        assert int(value) >= 0


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_coverage_to_bedgraph_depth_values(
    fixture, api: CoverageExporter, tmp_path, monkeypatch
):
    # Exercise the depth arithmetic directly against a small synthetic
    # dataset, since the simulated fixture data fills call_AD with a
    # missing-value sentinel throughout (see test above).
    contig = api.contigs[0]
    sample = str(api.sample_metadata()["sample_id"].iloc[0])

    ds = xr.Dataset(
        data_vars={
            "call_AD": (
                ["variants", "samples", "alleles"],
                da.from_array(
                    np.array(
                        [
                            [[10, 2, 0, 0]],  # total depth 12
                            [[-1, -1, -1, -1]],  # missing, should be skipped
                            [[5, -1, 0, 0]],  # negative components clipped to 0
                        ],
                        dtype="i2",
                    ),
                    chunks=(2, 1, 4),
                ),
            ),
        },
        coords={
            "variant_position": (
                "variants",
                da.from_array(np.array([100, 200, 300]), chunks=2),
            ),
            "variant_contig": (
                "variants",
                da.from_array(np.array([0, 0, 0], dtype="u1"), chunks=2),
            ),
            "sample_id": ("samples", np.array([sample])),
        },
        attrs={"contigs": (contig,)},
    )
    monkeypatch.setattr(api, "snp_calls", lambda **kwargs: ds)

    output_path = str(tmp_path / "test_snp_coverage_values.bedgraph")
    api.snp_calls_coverage_to_bedgraph(
        output_path=output_path,
        region=contig,
        sample=sample,
    )

    with open(output_path) as f:
        lines = f.readlines()

    data_lines = [line.rstrip("\n").split("\t") for line in lines[1:]]
    assert data_lines == [
        [contig, "99", "100", "12"],
        [contig, "299", "300", "5"],
    ]


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_calls_coverage_to_bedgraph_unknown_sample(
    fixture, api: CoverageExporter, tmp_path
):
    region = api.contigs[0]
    output_path = str(tmp_path / "test_snp_coverage_unknown.bedgraph")
    with pytest.raises(ValueError, match="No data found for sample"):
        api.snp_calls_coverage_to_bedgraph(
            output_path=output_path,
            region=region,
            sample="not_a_real_sample",
        )
