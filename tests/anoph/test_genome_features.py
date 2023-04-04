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
    )


def case_ag3(ag3_fixture, ag3_api):
    return ag3_fixture, ag3_api


def case_af1(af1_fixture, af1_api):
    return af1_fixture, af1_api


@parametrize_with_cases("fixture,api", cases=".")
def test_genome_features_no_attributes(fixture, api):
    df_gf = api.genome_features(attributes=None)
    assert isinstance(df_gf, pd.DataFrame)
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
    expected_cols = gff3_cols + ["attributes"]
    assert df_gf.columns.to_list() == expected_cols
    assert len(df_gf) > 0
    for contig in df_gf["contig"].unique():
        assert contig in fixture.contigs


# TODO more tests
