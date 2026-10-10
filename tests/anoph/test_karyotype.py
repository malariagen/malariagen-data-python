import pandas as pd
import pytest

from malariagen_data import ag3 as _ag3
from malariagen_data import af1 as _af1
from malariagen_data.anoph.karyotype import AnophelesKaryotypeAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesKaryotypeAnalysis(
        url=ag3_sim_fixture.url,
        public_url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        major_version_number=_ag3.MAJOR_VERSION_NUMBER,
        major_version_path=_ag3.MAJOR_VERSION_PATH,
        pre=True,
        gff_gene_type="gene",
        gff_gene_name_attribute="Name",
        gff_default_attributes=("ID", "Parent", "Name", "description"),
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesKaryotypeAnalysis(
        url=af1_sim_fixture.url,
        public_url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=True,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
    )


def test_load_inversion_tags(ag3_sim_api):
    df = ag3_sim_api.load_inversion_tags(inversion="2Rb")
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"inversion", "contig", "position", "alt_allele"}
    assert (df["inversion"] == "2Rb").all()
    assert (df["contig"] == "2R").all()
    assert len(df) > 0


def test_load_inversion_tags_2la(ag3_sim_api):
    df = ag3_sim_api.load_inversion_tags(inversion="2La")
    assert isinstance(df, pd.DataFrame)
    assert (df["inversion"] == "2La").all()
    assert (df["contig"] == "2L").all()
    assert len(df) > 0


def test_load_inversion_tags_invalid(ag3_sim_api):
    with pytest.raises(ValueError, match="Unknown inversion"):
        ag3_sim_api.load_inversion_tags(inversion="X_x")


def test_load_inversion_tags_not_implemented(af1_sim_api):
    with pytest.raises(NotImplementedError):
        af1_sim_api.load_inversion_tags(inversion="2La")


def test_karyotype(ag3_sim_api):
    df = ag3_sim_api.karyotype(inversion="2Rb")
    assert isinstance(df, pd.DataFrame)
    expected_cols = {
        "sample_id",
        "inversion",
        "karyotype_2Rb_mean",
        "karyotype_2Rb",
        "total_tag_snps",
    }
    assert set(df.columns) == expected_cols
    assert (df["inversion"] == "2Rb").all()
    assert all(df["karyotype_2Rb"].isin([0, 1, 2]))
    assert all(df["karyotype_2Rb_mean"].between(0, 2))


def test_karyotype_invalid_inversion(ag3_sim_api):
    with pytest.raises(ValueError, match="Unknown inversion"):
        ag3_sim_api.karyotype(inversion="X_x")


def test_karyotype_not_implemented(af1_sim_api):
    with pytest.raises(NotImplementedError):
        af1_sim_api.karyotype(inversion="2La")
