import pytest
import pandas as pd
from malariagen_data.anoph.snp_frq import AnophelesSnpFrequencyAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    from malariagen_data import ag3 as _ag3

    return AnophelesSnpFrequencyAnalysis(
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
    )


def test_snp_allele_frequencies_max_af_nan(ag3_sim_api):
    df_gff = ag3_sim_api.genome_features(attributes=["ID"])
    transcript = df_gff.query("type == 'mRNA'")["ID"].iloc[0]

    # Use a small min_cohort_size so we don't get ValueError.
    df = ag3_sim_api.snp_allele_frequencies(
        transcript=transcript,
        cohorts="admin1_year",
        min_cohort_size=1,
        drop_invariant=False,
    )

    assert isinstance(df, pd.DataFrame)
    if "max_af" in df.columns:
        assert not df["max_af"].isna().any(), "max_af contains NaNs"
        # Check that it's float
        assert df["max_af"].dtype.kind in "fc"  # float or complex


def test_aa_allele_frequencies_max_af_nan(ag3_sim_api):
    df_gff = ag3_sim_api.genome_features(attributes=["ID"])
    transcript = df_gff.query("type == 'mRNA'")["ID"].iloc[0]

    df = ag3_sim_api.aa_allele_frequencies(
        transcript=transcript,
        cohorts="admin1_year",
        min_cohort_size=1,
        drop_invariant=False,
    )

    if len(df) > 0:
        assert not df["max_af"].isna().any(), "max_af contains NaNs in AA frequencies"
        assert df["max_af"].dtype.kind in "fc"
