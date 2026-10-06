from types import SimpleNamespace

import igv_notebook  # type: ignore
import numpy as np
import pandas as pd
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
import malariagen_data.anoph.igv as igv_module
from malariagen_data.anoph.igv import AnophelesIgv


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
def af1_sim_api(af1_sim_fixture):
    return AnophelesIgv(
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
def test_igv(fixture, api: AnophelesIgv):
    region = fixture.random_region_str()
    browser = api.igv(region=region, init=False)
    assert isinstance(browser, igv_notebook.Browser)


@parametrize_with_cases("fixture,api", cases=".")
def test_view_alignments(fixture, api: AnophelesIgv):
    region = fixture.random_region_str()
    sample = str(np.random.choice(api.sample_metadata()["sample_id"]))
    ret = api.view_alignments(region=region, sample=sample, init=False)
    # No return value to avoid cluttering notebook output.
    assert ret is None


def test_view_alignments_with_local_snp_vcf(monkeypatch):
    class FakeAnophelesIgv(AnophelesIgv):
        @property
        def site_mask_ids(self):
            return []

        def sample_metadata(self):
            return pd.DataFrame({"sample_id": ["S1"], "sample_set": ["set1"]})

        def wgs_data_catalog(self, sample_set):
            return pd.DataFrame(
                {
                    "sample_id": ["S1"],
                    "alignments_bam": ["https://example.org/S1.bam"],
                    "snp_genotypes_vcf": ["https://example.org/S1.vcf.gz"],
                }
            )

    api = object.__new__(FakeAnophelesIgv)
    captured = {}
    monkeypatch.setattr(
        igv_module,
        "_parse_single_region",
        lambda self, region: SimpleNamespace(contig="2L"),
    )
    monkeypatch.setattr(
        api,
        "igv",
        lambda **kwargs: captured.update(kwargs),
    )
    local_vcf_url = "/data/sample.vcf.gz"
    api.view_alignments(
        region="2L:1-10",
        sample="S1",
        init=False,
        snp_vcf_url=local_vcf_url,
    )

    snp_track = next(track for track in captured["tracks"] if track["name"] == "SNPs")
    assert snp_track["url"] == local_vcf_url
    # A local VCF has no index, so IGV must load it in full.
    assert snp_track["indexed"] is False
    assert "indexURL" not in snp_track

    captured.clear()
    api.view_alignments(region="2L:1-10", sample="S1", init=False)
    snp_track = next(track for track in captured["tracks"] if track["name"] == "SNPs")
    assert snp_track["url"] == "https://example.org/S1.vcf.gz"
    assert snp_track["indexURL"] == "https://example.org/S1.vcf.gz.tbi"
    assert "indexed" not in snp_track
