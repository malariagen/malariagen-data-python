import random

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
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


def random_transcript(*, api):
    df_gff = api.genome_features(attributes=["ID", "Parent"])
    df_transcripts = df_gff.query("type == 'mRNA'")
    transcript_ids = df_transcripts["ID"].dropna().to_list()
    transcript_id = random.choice(transcript_ids)
    transcript = df_transcripts.set_index("ID").loc[transcript_id]
    return transcript


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_effects(fixture, api: AnophelesSnpFrequencyAnalysis):
    # Pick a random transcript.
    transcript = random_transcript(api=api)

    # Pick a random site mask.
    site_mask = random.choice(api.site_mask_ids + (None,))

    # Compute effects.
    df = api.snp_effects(transcript=transcript.name, site_mask=site_mask)
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
    assert np.all(df["contig"] == transcript["contig"])
    position = df["position"].values
    assert np.all(position >= transcript["start"])
    assert np.all(position <= transcript["end"])
    assert np.all(position[1:] >= position[:-1])
    expected_alleles = list("ACGT")
    assert np.all(df["ref_allele"].isin(expected_alleles))
    assert np.all(df["alt_allele"].isin(expected_alleles))
    assert np.all(df["transcript"] == transcript.name)
    assert np.all(df["effect"].isin(expected_effects))
    assert np.all(df["impact"].isin(expected_impacts))
    df_aa = df[~df["aa_change"].isna()]
    expected_aa_change = (
        df_aa["ref_aa"] + df_aa["aa_pos"].astype(int).astype(str) + df_aa["alt_aa"]
    )
    assert np.all(df_aa["aa_change"] == expected_aa_change)


def check_frequency(x):
    loc_nan = np.isnan(x)
    assert np.all(x[~loc_nan] >= 0)
    assert np.all(x[~loc_nan] <= 1)


def check_snp_allele_frequencies(
    *,
    api,
    df,
    cohort_labels,
    transcript,
):
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
    frq_fields = ["frq_" + s for s in cohort_labels] + ["max_af"]
    expected_fields = universal_fields + frq_fields + effects_fields
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
    assert np.all(df["contig"] == transcript["contig"])
    position = df["position"].values
    assert np.all(position >= transcript["start"])
    assert np.all(position <= transcript["end"])
    assert np.all(position[1:] >= position[:-1])
    assert np.all(df["ref_allele"].isin(expected_alleles))
    assert np.all(df["alt_allele"].isin(expected_alleles))
    assert np.all(df["transcript"] == transcript.name)
    assert np.all(df["effect"].isin(expected_effects))
    assert np.all(df["impact"].isin(expected_impacts))
    df_aa = df[~df["aa_change"].isna()]
    expected_aa_change = (
        df_aa["ref_aa"] + df_aa["aa_pos"].astype(int).astype(str) + df_aa["alt_aa"]
    )
    assert np.all(df_aa["aa_change"] == expected_aa_change)
    for f in frq_fields:
        x = df[f]
        check_frequency(x)


def check_aa_allele_frequencies(
    *,
    df,
    cohort_labels,
    transcript,
):
    assert isinstance(df, pd.DataFrame)

    # Check columns.
    universal_fields = [
        "label",
    ]
    effects_fields = [
        "transcript",
        "effect",
        "impact",
        "aa_pos",
        "ref_allele",
        "alt_allele",
        "ref_aa",
        "alt_aa",
    ]
    frq_fields = ["frq_" + s for s in cohort_labels] + ["max_af"]
    expected_fields = universal_fields + frq_fields + effects_fields
    expected_fields = universal_fields + frq_fields + effects_fields
    assert sorted(df.columns.tolist()) == sorted(expected_fields)
    assert df.index.names == [
        "aa_change",
        "contig",
        "position",
    ]

    # Check some values.
    df = df.reset_index()
    assert np.all(df["contig"] == transcript["contig"])
    position = df["position"].values
    assert np.all(position >= transcript["start"])
    assert np.all(position <= transcript["end"])
    assert np.all(position[1:] >= position[:-1])
    assert np.all(df["ref_allele"].isin(expected_alleles))
    # N.B., alt_allele may contain multiple alleles, e.g., "{A,T}", if
    # multiple SNP alleles at the same position cause the same amino acid
    # change.
    assert np.all(df["transcript"] == transcript.name)
    assert np.all(df["effect"].isin(expected_effects))
    assert np.all(df["impact"].isin(expected_impacts))
    df_aa = df[~df["aa_change"].isna()]
    expected_aa_change = (
        df_aa["ref_aa"] + df_aa["aa_pos"].astype(int).astype(str) + df_aa["alt_aa"]
    )
    assert np.all(df_aa["aa_change"] == expected_aa_change)
    for f in frq_fields:
        x = df[f]
        check_frequency(x)


@pytest.mark.parametrize("cohorts", ["admin1_year", "admin2_month", "country"])
@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_with_str_cohorts(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
    cohorts,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice(api.site_mask_ids + (None,))
    min_cohort_size = random.randint(0, 10)
    transcript = random_transcript(api=api)

    # Figure out expected cohort labels.
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    if "cohort_" + cohorts in df_samples:
        cohort_column = "cohort_" + cohorts
    else:
        cohort_column = cohorts
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    # Set up call params.
    params = dict(
        transcript=transcript.name,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
        sample_sets=sample_sets,
        drop_invariant=True,
    )

    # Run the function under test.
    df_snp = api.snp_allele_frequencies(**params)

    # Standard checks.
    check_snp_allele_frequencies(
        api=api,
        df=df_snp,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )

    # Run the function under test.
    df_aa = api.aa_allele_frequencies(**params)

    # Standard checks.
    check_aa_allele_frequencies(
        df=df_aa,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_with_str_cohorts_and_sample_query(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    # Pick test parameters at random.
    sample_sets = None
    site_mask = random.choice(api.site_mask_ids + (None,))
    min_cohort_size = random.randint(0, 10)
    transcript = random_transcript(api=api)
    cohorts = random.choice(
        ["admin1_year", "admin1_month", "admin2_year", "admin2_month"]
    )
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    countries = df_samples["country"].unique()
    country = random.choice(countries)
    sample_query = f"country == '{country}'"

    # Figure out expected cohort labels.
    df_samples = api.sample_metadata(sample_sets=sample_sets, sample_query=sample_query)
    cohort_column = "cohort_" + cohorts
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    # Set up call params.
    params = dict(
        transcript=transcript.name,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
        sample_sets=sample_sets,
        sample_query=sample_query,
        drop_invariant=True,
    )

    # Run the function under test.
    df_snp = api.snp_allele_frequencies(**params)

    # Standard checks.
    check_snp_allele_frequencies(
        api=api,
        df=df_snp,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )

    # Run the function under test.
    df_aa = api.aa_allele_frequencies(**params)

    # Standard checks.
    check_aa_allele_frequencies(
        df=df_aa,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_with_dict_cohorts(
    fixture, api: AnophelesSnpFrequencyAnalysis
):
    # Pick test parameters at random.
    sample_sets = None  # all sample sets
    site_mask = random.choice(api.site_mask_ids + (None,))
    min_cohort_size = random.randint(0, 10)
    transcript = random_transcript(api=api)

    # Create cohorts by country.
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    cohort_counts = df_samples["country"].value_counts()
    cohorts = {cohort: f"country == '{cohort}'" for cohort in cohort_counts.index}
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    # Set up call params.
    params = dict(
        transcript=transcript.name,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
        sample_sets=sample_sets,
        drop_invariant=True,
    )

    # Run the function under test.
    df_snp = api.snp_allele_frequencies(**params)

    # Standard checks.
    check_snp_allele_frequencies(
        api=api,
        df=df_snp,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )

    # Run the function under test.
    df_aa = api.aa_allele_frequencies(**params)

    # Standard checks.
    check_aa_allele_frequencies(
        df=df_aa,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_without_drop_invariant(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice(api.site_mask_ids + (None,))
    min_cohort_size = random.randint(0, 10)
    transcript = random_transcript(api=api)
    cohorts = random.choice(["admin1_year", "admin2_month", "country"])

    # Figure out expected cohort labels.
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    if "cohort_" + cohorts in df_samples:
        cohort_column = "cohort_" + cohorts
    else:
        cohort_column = cohorts
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    # Set up call params.
    params = dict(
        transcript=transcript.name,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
        sample_sets=sample_sets,
    )

    # Run the function under test.
    df_snp_a = api.snp_allele_frequencies(drop_invariant=True, **params)
    df_snp_b = api.snp_allele_frequencies(drop_invariant=False, **params)

    # Standard checks.
    check_snp_allele_frequencies(
        api=api,
        df=df_snp_a,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )
    check_snp_allele_frequencies(
        api=api,
        df=df_snp_b,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )

    # Check specifics.
    assert len(df_snp_b) > len(df_snp_a)


@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_without_effects(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice(api.site_mask_ids + (None,))
    min_cohort_size = random.randint(0, 10)
    transcript = random_transcript(api=api)
    cohorts = random.choice(["admin1_year", "admin2_month", "country"])

    # Figure out expected cohort labels.
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    if "cohort_" + cohorts in df_samples:
        cohort_column = "cohort_" + cohorts
    else:
        cohort_column = cohorts
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    # Set up call params.
    params = dict(
        transcript=transcript.name,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
        sample_sets=sample_sets,
        drop_invariant=True,
    )

    # Run the function under test.
    df_snp_a = api.snp_allele_frequencies(effects=True, **params)
    df_snp_b = api.snp_allele_frequencies(effects=False, **params)

    # Standard checks.
    check_snp_allele_frequencies(
        api=api,
        df=df_snp_a,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )

    # Check specifics.
    assert len(df_snp_b) == len(df_snp_a)

    # Check columns and index names.
    filter_fields = [f"pass_{m}" for m in api.site_mask_ids]
    universal_fields = filter_fields + ["label"]
    frq_fields = ["frq_" + s for s in cohort_labels] + ["max_af"]
    expected_fields = universal_fields + frq_fields
    assert sorted(df_snp_b.columns.tolist()) == sorted(expected_fields)
    assert df_snp_b.index.names == [
        "contig",
        "position",
        "ref_allele",
        "alt_allele",
    ]

    # Compare values with and without effects.
    comparable_fields = (
        [
            "contig",
            "position",
            "ref_allele",
            "alt_allele",
        ]
        + filter_fields
        + frq_fields
    )
    # N.B., values of the "label" field are different with and without
    # effects, so don't compare them.
    assert_frame_equal(
        df_snp_b.reset_index()[comparable_fields],
        df_snp_a.reset_index()[comparable_fields],
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_with_bad_transcript(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    site_mask = random.choice(api.site_mask_ids + (None,))
    min_cohort_size = random.randint(0, 10)
    cohorts = random.choice(["admin1_year", "admin2_month", "country"])

    # Set up call params.
    params = dict(
        transcript="foobar",
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
        sample_sets=sample_sets,
        drop_invariant=True,
    )

    # Run the function under test.
    with pytest.raises(ValueError):
        api.snp_allele_frequencies(**params)


@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_with_dup_samples(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = random.choice(all_sample_sets)
    site_mask = random.choice(api.site_mask_ids + (None,))
    min_cohort_size = random.randint(0, 10)
    transcript = random_transcript(api=api)
    cohorts = random.choice(["admin1_year", "admin2_month", "country"])

    # Set up call params.
    params = dict(
        transcript=transcript.name,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
    )

    # Run the function under test.
    df_snp_a = api.snp_allele_frequencies(sample_sets=[sample_set], **params)
    df_snp_b = api.snp_allele_frequencies(
        sample_sets=[sample_set, sample_set], **params
    )

    # Expect automatically deduplicate sample sets.
    assert_frame_equal(df_snp_b, df_snp_a)

    # Run the function under test.
    df_aa_a = api.aa_allele_frequencies(sample_sets=[sample_set], **params)
    df_aa_b = api.aa_allele_frequencies(sample_sets=[sample_set, sample_set], **params)

    # Expect automatically deduplicate sample sets.
    assert_frame_equal(df_aa_b, df_aa_a)


# from test_anopheles
# TODO: test_snp_allele_frequencies_advanced_with_dup_samples
# TODO: test_aa_allele_frequencies_advanced_with_dup_samples

# from test_ag3 and test_af1
# TODO: test_allele_frequencies_advanced_with_transcript
# TODO: test_allele_frequencies_advanced_with_area_by
# TODO: test_allele_frequencies_advanced_with_period_by
# TODO: test_allele_frequencies_advanced_with_sample_sets
# TODO: test_allele_frequencies_advanced_with_sample_query
# TODO: test_allele_frequencies_advanced_with_min_cohort_size
# TODO: test_allele_frequencies_advanced_with_variant_query
# TODO: test_allele_frequencies_advanced_with_nobs_mode
