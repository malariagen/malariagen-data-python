import random

import numpy as np
import pandas as pd
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.snp_frq import AnophelesSnpFrequencyAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesSnpFrequencyAnalysis(
        url=ag3_sim_fixture.url,
        config_path=_ag3.CONFIG_PATH,
        gcs_url=_ag3.GCS_URL,
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
        gff_default_attributes=("ID", "Parent", "Name", "description"),
        default_site_mask="gamb_colu_arab",
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_ag3.TAXON_COLORS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesSnpFrequencyAnalysis(
        url=af1_sim_fixture.url,
        config_path=_af1.CONFIG_PATH,
        gcs_url=_af1.GCS_URL,
        major_version_number=_af1.MAJOR_VERSION_NUMBER,
        major_version_path=_af1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        default_site_mask="funestus",
        results_cache=af1_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_af1.TAXON_COLORS,
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


expected_alleles = list("ACGT")
expected_effects = [
    "FIVE_PRIME_UTR",
    "THREE_PRIME_UTR",
    "SYNONYMOUS_CODING",
    "NON_SYNONYMOUS_CODING",
    "START_LOST",
    "STOP_LOST",
    "STOP_GAINED",
    "SPLICE_CORE",
    "SPLICE_REGION",
    "INTRONIC",
    "TRANSCRIPT",
]
expected_impacts = [
    "HIGH",
    "MODERATE",
    "LOW",
    "MODIFIER",
]


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_effects(fixture, api: AnophelesSnpFrequencyAnalysis):
    # Pick a random transcript.
    df_gff = api.genome_features(attributes=["ID", "Parent"])
    df_transcripts = df_gff.query("type == 'mRNA'")
    transcripts = df_transcripts["ID"].dropna().to_list()
    transcript = random.choice(transcripts)
    transcript_rec = df_transcripts.set_index("ID").loc[transcript]

    # Pick a random site mask.
    site_mask = random.choice(api.site_mask_ids + (None,))

    # Compute effects.
    df = api.snp_effects(transcript=transcript, site_mask=site_mask)
    assert isinstance(df, pd.DataFrame)

    # Check columns.
    expected_fields = (
        [
            "contig",
            "position",
            "ref_allele",
            "alt_allele",
        ]
        + [f"pass_{m}" for m in api.site_mask_ids]
        + [
            "transcript",
            "effect",
            "impact",
            "ref_codon",
            "alt_codon",
            "aa_pos",
            "ref_aa",
            "alt_aa",
            "aa_change",
        ]
    )
    assert df.columns.tolist() == expected_fields

    # Check some values.
    assert np.all(df["contig"] == transcript_rec["contig"])
    position = df["position"].values
    assert np.all(position >= transcript_rec["start"])
    assert np.all(position <= transcript_rec["end"])
    assert np.all(position[1:] >= position[:-1])
    expected_alleles = list("ACGT")
    assert np.all(df["ref_allele"].isin(expected_alleles))
    assert np.all(df["alt_allele"].isin(expected_alleles))
    assert np.all(df["transcript"] == transcript)
    assert np.all(df["effect"].isin(expected_effects))
    assert np.all(df["impact"].isin(expected_impacts))
    df_aa = df[~df["aa_change"].isna()]
    expected_aa_change = (
        df_aa["ref_aa"] + df_aa["aa_pos"].astype(int).astype(str) + df_aa["alt_aa"]
    )
    assert np.all(df_aa["aa_change"] == expected_aa_change)


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_allele_frequencies_with_str_cohorts(
    fixture, api: AnophelesSnpFrequencyAnalysis
):
    # Pick test parameters at random.
    cohorts = random.choice(
        [
            "admin1_year",
            "admin1_month",
            "admin2_year",
            "admin2_month",
        ]
    )
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = random.choice(all_sample_sets)
    site_mask = random.choice(api.site_mask_ids + (None,))
    min_cohort_size = random.randint(0, 10)
    df_gff = api.genome_features(attributes=["ID", "Parent"])
    df_transcripts = df_gff.query("type == 'mRNA'")
    transcripts = df_transcripts["ID"].dropna().to_list()
    transcript = random.choice(transcripts)
    transcript_rec = df_transcripts.set_index("ID").loc[transcript]

    # Run the function under test.
    df = api.snp_allele_frequencies(
        transcript=transcript,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
        sample_sets=sample_set,
        drop_invariant=True,
        effects=True,
    )
    assert isinstance(df, pd.DataFrame)

    # Check columns.
    universal_fields = [f"pass_{m}" for m in api.site_mask_ids] + [
        "label",
    ]
    effects_fields = [
        "transcript",
        "effect",
        "impact",
        "ref_codon",
        "alt_codon",
        "aa_pos",
        "ref_aa",
        "alt_aa",
    ]
    df_coh = api.cohorts_metadata(sample_sets=sample_set)
    coh_nm = "cohort_" + cohorts
    coh_counts = df_coh[coh_nm].dropna().value_counts()
    cohort_labels = coh_counts[coh_counts >= min_cohort_size].index.to_list()
    frq_cohort_labels = ["frq_" + s for s in cohort_labels]
    expected_fields = universal_fields + frq_cohort_labels + ["max_af"] + effects_fields
    assert sorted(df.columns.tolist()) == sorted(expected_fields)
    assert df.index.names == [
        "contig",
        "position",
        "ref_allele",
        "alt_allele",
        "aa_change",
    ]

    # Check some values.
    df = df.reset_index()
    assert np.all(df["contig"] == transcript_rec["contig"])
    position = df["position"].values
    assert np.all(position >= transcript_rec["start"])
    assert np.all(position <= transcript_rec["end"])
    assert np.all(position[1:] >= position[:-1])
    assert np.all(df["ref_allele"].isin(expected_alleles))
    assert np.all(df["alt_allele"].isin(expected_alleles))
    assert np.all(df["transcript"] == transcript)
    assert np.all(df["effect"].isin(expected_effects))
    assert np.all(df["impact"].isin(expected_impacts))
    df_aa = df[~df["aa_change"].isna()]
    expected_aa_change = (
        df_aa["ref_aa"] + df_aa["aa_pos"].astype(int).astype(str) + df_aa["alt_aa"]
    )
    assert np.all(df_aa["aa_change"] == expected_aa_change)


# from test_ag3 and test_af1
# TODO: test_snp_allele_frequencies_with_dict_cohorts
# TODO: test_snp_allele_frequencies_with_sample_query
# TODO: test_aa_allele_frequencies_with_str_cohorts
# TODO: test_allele_frequencies_advanced_with_transcript
# TODO: test_allele_frequencies_advanced_with_area_by
# TODO: test_allele_frequencies_advanced_with_period_by
# TODO: test_allele_frequencies_advanced_with_sample_sets
# TODO: test_allele_frequencies_advanced_with_sample_query
# TODO: test_allele_frequencies_advanced_with_min_cohort_size
# TODO: test_allele_frequencies_advanced_with_variant_query
# TODO: test_allele_frequencies_advanced_with_nobs_mode

# from test_anopheles
# TODO: test_snp_allele_frequencies_with_str_cohorts
# TODO: test_snp_allele_frequencies_with_dup_samples
# TODO: test_snp_allele_frequencies_with_bad_transcript
# TODO: test_aa_allele_frequencies_with_dup_samples
# TODO: test_snp_allele_frequencies_advanced_with_dup_samples
# TODO: test_aa_allele_frequencies_advanced_with_dup_samples
