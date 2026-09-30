import numpy as np
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.igv import AnophelesIgv
from malariagen_data.anoph.to_coverage import CoverageExporter


class _IgvWithCoverage(CoverageExporter, AnophelesIgv):
    """Combines AnophelesIgv with CoverageExporter, the way the
    top-level Ag3/Af1 data resource classes do, so that the local
    coverage fallback functionality can be exercised in isolation."""


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesIgv(
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
    )


@pytest.fixture
def ag3_sim_api_with_coverage(ag3_sim_fixture):
    return _IgvWithCoverage(
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
        default_coverage_calls_analysis="gamb_colu",
        discordant_read_calls_analysis=ag3_sim_fixture.config[
            "DEFAULT_DISCORDANT_READ_CALLS_ANALYSIS"
        ],
    )


@pytest.fixture
def af1_sim_api_with_coverage(af1_sim_fixture):
    return _IgvWithCoverage(
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
        default_coverage_calls_analysis="funestus",
        discordant_read_calls_analysis=None,
    )


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api_with_coverage):
    return ag3_sim_fixture, ag3_sim_api_with_coverage


def case_af1_sim(af1_sim_fixture, af1_sim_api_with_coverage):
    return af1_sim_fixture, af1_sim_api_with_coverage


@parametrize_with_cases("fixture,api", cases=".")
def test_view_coverage(fixture, api: _IgvWithCoverage):
    region = api.contigs[0]
    sample = str(np.random.choice(api.sample_metadata()["sample_id"]))
    ret = api.view_coverage(region=region, sample=sample, init=False)
    # No return value to avoid cluttering notebook output.
    assert ret is None


@parametrize_with_cases("fixture,api", cases=".")
def test_view_alignments_with_local_coverage_fallback(fixture, api: _IgvWithCoverage):
    region = api.contigs[0]
    sample = str(np.random.choice(api.sample_metadata()["sample_id"]))
    ret = api.view_alignments(
        region=region,
        sample=sample,
        init=False,
        local_coverage_fallback=True,
    )
    assert ret is None


@parametrize_with_cases("fixture,api", cases=".")
def test_view_coverage_writes_bedgraph(fixture, api: _IgvWithCoverage, tmp_path):
    region = api.contigs[0]
    sample = str(np.random.choice(api.sample_metadata()["sample_id"]))
    output_path = str(tmp_path / "coverage.bedgraph")
    api.view_coverage(region=region, sample=sample, init=False, output_path=output_path)
    with open(output_path) as f:
        first_line = f.readline()
    assert first_line.startswith("track type=bedGraph")


def test_local_coverage_fallback_requires_cnv_mixin(ag3_sim_fixture, ag3_sim_api):
    # Plain AnophelesIgv, without CoverageExporter mixed in, should
    # raise a clear error rather than an AttributeError if asked for a
    # local coverage track.
    region = ag3_sim_api.contigs[0]
    sample = str(np.random.choice(ag3_sim_api.sample_metadata()["sample_id"]))
    with pytest.raises(NotImplementedError):
        ag3_sim_api.view_alignments(
            region=region,
            sample=sample,
            init=False,
            local_coverage_fallback=True,
        )
