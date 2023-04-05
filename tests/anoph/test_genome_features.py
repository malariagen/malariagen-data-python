import random

import pandas as pd
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.genome_features import AnophelesGenomeFeaturesData


@pytest.fixture
def ag3_api(ag3_fixture):
    return AnophelesGenomeFeaturesData(
        url=ag3_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        gcs_url=_ag3.GCS_URL,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
        gff_gene_type="gene",
        gff_default_attributes=("ID", "Parent", "Name", "description"),
    )


@pytest.fixture
def af1_api(af1_fixture):
    return AnophelesGenomeFeaturesData(
        url=af1_fixture.url,
        config_path=_af1.CONFIG_PATH,
        gcs_url=_af1.GCS_URL,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
    )


def case_ag3(ag3_fixture, ag3_api):
    return ag3_fixture, ag3_api


def case_af1(af1_fixture, af1_api):
    return af1_fixture, af1_api


gff3_cols = [
    "contig",
    "source",
    "type",
    "start",
    "end",
    "score",
    "strand",
    "phase",
]


@parametrize_with_cases("fixture,api", cases=".")
def test_genome_features_no_attributes(fixture, api):
    df_gf = api.genome_features(attributes=None)
    assert isinstance(df_gf, pd.DataFrame)
    expected_cols = gff3_cols + ["attributes"]
    assert df_gf.columns.to_list() == expected_cols
    assert len(df_gf) > 0
    for contig in df_gf["contig"].unique():
        assert contig in fixture.contigs


def test_genome_features_default_attributes_ag3(ag3_api):
    df_gf = ag3_api.genome_features()
    assert isinstance(df_gf, pd.DataFrame)
    expected_cols = gff3_cols + ["ID", "Parent", "Name", "description"]
    assert df_gf.columns.to_list() == expected_cols


def test_genome_features_default_attributes_af1(af1_api):
    df_gf = af1_api.genome_features()
    assert isinstance(df_gf, pd.DataFrame)
    expected_cols = gff3_cols + ["ID", "Parent", "Note", "description"]
    assert df_gf.columns.to_list() == expected_cols


@parametrize_with_cases("fixture,api", cases=".")
def test_genome_features_region_contig(fixture, api):
    for contig in fixture.contigs:
        df_gf = api.genome_features(region=contig, attributes=None)
        expected_cols = gff3_cols + ["attributes"]
        assert df_gf.columns.to_list() == expected_cols
        assert len(df_gf) > 0
        assert (df_gf["contig"] == contig).all()


@parametrize_with_cases("fixture,api", cases=".")
def test_genome_features_region_string(fixture, api):
    for contig in fixture.contigs:
        contig_size = fixture.contig_sizes[contig]
        region_start = random.randint(1, contig_size)
        region_end = random.randint(region_start, contig_size)
        region = f"{contig}:{region_start:,}-{region_end:,}"
        df_gf = api.genome_features(region=region, attributes=None)
        expected_cols = gff3_cols + ["attributes"]
        assert df_gf.columns.to_list() == expected_cols
        # N.B., it's possible that the region overlaps no features.
        if len(df_gf) > 0:
            assert (df_gf["contig"] == contig).all()
            assert (df_gf["end"] >= region_start).all()
            assert (df_gf["start"] <= region_end).all()
