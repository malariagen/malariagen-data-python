import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from pytest_cases import parametrize_with_cases
import xarray as xr
from numpy.testing import assert_allclose, assert_array_equal
from .conftest import Af1Simulator, Ag3Simulator

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.snp_frq import AnophelesSnpFrequencyAnalysis
from malariagen_data.util import compare_series_like
from .test_frq import (
    check_plot_frequencies_heatmap,
    check_plot_frequencies_time_series,
    check_plot_frequencies_time_series_with_taxa,
    check_plot_frequencies_time_series_with_areas,
    check_plot_frequencies_interactive_map,
)


rng = np.random.default_rng(seed=42)


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
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


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesSnpFrequencyAnalysis(
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
    transcript_id = rng.choice(transcript_ids)
    transcript = df_transcripts.set_index("ID").loc[transcript_id]
    return transcript


@parametrize_with_cases("fixture,api", cases=".")
def test_snp_effects(fixture, api: AnophelesSnpFrequencyAnalysis):
    # Pick a random transcript.
    transcript = random_transcript(api=api)

    # Pick a random site mask.
    if isinstance(fixture, Af1Simulator):
        valid_site_masks = ["funestus"]
    elif isinstance(fixture, Ag3Simulator):
        valid_site_masks = ["gamb_colu_arab", "gamb_colu", "arab"]
    else:
        valid_site_masks = [""] + list(api.site_mask_ids)
    site_mask = rng.choice(valid_site_masks)

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
    position = df["position"].to_numpy()
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


@pytest.mark.parametrize(
    "cohorts", ["admin1_year", "admin2_month", "country", "foobar"]
)
@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_with_str_cohorts(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
    cohorts,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = rng.choice(all_sample_sets)
    if isinstance(fixture, Af1Simulator):
        valid_site_masks = ["funestus"]
    elif isinstance(fixture, Ag3Simulator):
        valid_site_masks = ["gamb_colu_arab", "gamb_colu", "arab"]
    else:
        valid_site_masks = [""] + list(api.site_mask_ids)
    site_mask = rng.choice(valid_site_masks)

    min_cohort_size = int(rng.integers(0, 2))
    transcript = random_transcript(api=api)

    # Set up call params.
    params = dict(
        transcript=transcript.name,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
        sample_sets=sample_sets,
        drop_invariant=True,
    )

    # Test behaviour with bad cohorts param.
    if cohorts == "foobar":
        with pytest.raises(ValueError):
            api.snp_allele_frequencies(**params)
        return

    # Run the function under test.
    df_snp = api.snp_allele_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_snp)

    # Figure out expected cohort labels.
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    if "cohort_" + cohorts in df_samples:
        cohort_column = "cohort_" + cohorts
    else:
        cohort_column = cohorts
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    # Standard checks.
    check_snp_allele_frequencies(
        api=api,
        df=df_snp,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )

    # Run the function under test.
    df_aa = api.aa_allele_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_aa)

    # Standard checks.
    check_aa_allele_frequencies(
        df=df_aa,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )


@pytest.mark.parametrize("min_cohort_size", [0, 10, 100])
@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_with_min_cohort_size(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
    min_cohort_size,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = rng.choice(all_sample_sets)
    if isinstance(fixture, Af1Simulator):
        valid_site_masks = ["funestus"]
    elif isinstance(fixture, Ag3Simulator):
        valid_site_masks = ["gamb_colu_arab", "gamb_colu", "arab"]
    else:
        valid_site_masks = [""] + list(api.site_mask_ids)
    site_mask = rng.choice(valid_site_masks)
    transcript = random_transcript(api=api)
    cohorts = "admin1_year"

    # Set up call params.
    params = dict(
        transcript=transcript.name,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
        sample_sets=sample_sets,
        drop_invariant=True,
    )

    # Figure out expected cohort labels.
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    if "cohort_" + cohorts in df_samples:
        cohort_column = "cohort_" + cohorts
    else:
        cohort_column = cohorts
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    if len(cohort_labels) == 0:
        # No cohorts, expect error.
        with pytest.raises(ValueError):
            api.snp_allele_frequencies(**params)
        with pytest.raises(ValueError):
            api.aa_allele_frequencies(**params)
        return

    # Run the function under test.
    df_snp = api.snp_allele_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_snp)

    # Standard checks.
    check_snp_allele_frequencies(
        api=api,
        df=df_snp,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )

    # Run the function under test.
    df_aa = api.aa_allele_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_aa)

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
    if isinstance(fixture, Af1Simulator):
        valid_site_masks = ["funestus"]
    elif isinstance(fixture, Ag3Simulator):
        valid_site_masks = ["gamb_colu_arab", "gamb_colu", "arab"]
    else:
        valid_site_masks = [""] + list(api.site_mask_ids)
    site_mask = rng.choice(valid_site_masks)
    min_cohort_size = 0
    transcript = random_transcript(api=api)
    cohorts = rng.choice(["admin1_year", "admin1_month", "admin2_year", "admin2_month"])
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    countries = df_samples["country"].unique()
    country = rng.choice(countries)
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

    check_plot_frequencies_heatmap(api, df_snp)

    # Standard checks.
    check_snp_allele_frequencies(
        api=api,
        df=df_snp,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )

    # Run the function under test.
    df_aa = api.aa_allele_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_aa)

    # Standard checks.
    check_aa_allele_frequencies(
        df=df_aa,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_with_str_cohorts_and_sample_query_options(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    # Pick test parameters at random.
    sample_sets = None
    if isinstance(fixture, Af1Simulator):
        valid_site_masks = ["funestus"]
    elif isinstance(fixture, Ag3Simulator):
        valid_site_masks = ["gamb_colu_arab", "gamb_colu", "arab"]
    else:
        valid_site_masks = [""] + list(api.site_mask_ids)
    site_mask = rng.choice(valid_site_masks)
    min_cohort_size = 0
    transcript = random_transcript(api=api)
    cohorts = rng.choice(["admin1_year", "admin1_month", "admin2_year", "admin2_month"])
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    countries = df_samples["country"].unique().tolist()
    countries_list = rng.choice(countries, 2, replace=False).tolist()
    sample_query_options = {
        "local_dict": {
            "countries_list": countries_list,
        }
    }
    sample_query = "country in @countries_list"

    # Figure out expected cohort labels.
    df_samples = api.sample_metadata(
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
    )
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
        sample_query_options=sample_query_options,
        drop_invariant=True,
    )

    # Run the function under test.
    df_snp = api.snp_allele_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_snp)

    # Standard checks.
    check_snp_allele_frequencies(
        api=api,
        df=df_snp,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )

    # Run the function under test.
    df_aa = api.aa_allele_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_aa)

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
    site_mask = rng.choice(list(api.site_mask_ids) + [""])
    min_cohort_size = int(rng.integers(0, 2))
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

    check_plot_frequencies_heatmap(api, df_snp)

    # Standard checks.
    check_snp_allele_frequencies(
        api=api,
        df=df_snp,
        cohort_labels=cohort_labels,
        transcript=transcript,
    )

    # Run the function under test.
    df_aa = api.aa_allele_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_aa)

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
    sample_sets = rng.choice(all_sample_sets)
    site_mask = rng.choice(list(api.site_mask_ids) + [""])
    min_cohort_size = int(rng.integers(0, 2))
    transcript = random_transcript(api=api)
    cohorts = rng.choice(["admin1_year", "admin2_month", "country"])

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

    check_plot_frequencies_heatmap(api, df_snp_a)
    check_plot_frequencies_heatmap(api, df_snp_b)

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
    sample_sets = rng.choice(all_sample_sets)
    if isinstance(fixture, Af1Simulator):
        valid_site_masks = ["funestus"]
    elif isinstance(fixture, Ag3Simulator):
        valid_site_masks = ["gamb_colu_arab", "gamb_colu", "arab"]
    else:
        valid_site_masks = [""] + list(api.site_mask_ids)
    site_mask = rng.choice(valid_site_masks)
    min_cohort_size = int(rng.integers(0, 2))
    transcript = random_transcript(api=api)
    cohorts = rng.choice(["admin1_year", "admin2_month", "country"])

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

    check_plot_frequencies_heatmap(api, df_snp_a)
    check_plot_frequencies_heatmap(api, df_snp_b)

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
    sample_sets = rng.choice(all_sample_sets)
    site_mask = rng.choice(list(api.site_mask_ids) + [""])
    min_cohort_size = int(rng.integers(0, 2))
    cohorts = rng.choice(["admin1_year", "admin2_month", "country"])

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
def test_allele_frequencies_with_region(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = rng.choice(all_sample_sets)
    if isinstance(fixture, Af1Simulator):
        valid_site_masks = ["funestus"]
    elif isinstance(fixture, Ag3Simulator):
        valid_site_masks = ["gamb_colu_arab", "gamb_colu", "arab"]
    else:
        valid_site_masks = [""] + list(api.site_mask_ids)
    site_mask = rng.choice(valid_site_masks)
    min_cohort_size = int(rng.integers(0, 2))
    cohorts = rng.choice(["admin1_year", "admin2_month", "country"])
    # This should work, as long as effects=False - i.e., can get frequencies
    # for any genome region.
    transcript = fixture.random_region_str(region_size=500)

    # Set up call params.
    params = dict(
        transcript=transcript,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        site_mask=site_mask,
        sample_sets=sample_sets,
        drop_invariant=False,
        effects=False,
    )

    # Run the function under test.
    df_snp = api.snp_allele_frequencies(**params)

    check_plot_frequencies_heatmap(api, df_snp)

    # Basic checks.
    assert isinstance(df_snp, pd.DataFrame)
    assert len(df_snp) > 0

    # Figure out expected cohort labels.
    df_samples = api.sample_metadata(sample_sets=sample_sets)
    if "cohort_" + cohorts in df_samples:
        cohort_column = "cohort_" + cohorts
    else:
        cohort_column = cohorts
    cohort_counts = df_samples[cohort_column].value_counts()
    cohort_labels = cohort_counts[cohort_counts >= min_cohort_size].index.to_list()

    # Check columns and index names.
    filter_fields = [f"pass_{m}" for m in api.site_mask_ids]
    universal_fields = filter_fields + ["label"]
    frq_fields = ["frq_" + s for s in cohort_labels] + ["max_af"]
    expected_fields = universal_fields + frq_fields
    assert sorted(df_snp.columns.tolist()) == sorted(expected_fields)
    assert df_snp.index.names == [
        "contig",
        "position",
        "ref_allele",
        "alt_allele",
    ]


@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_with_dup_samples(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = rng.choice(all_sample_sets)
    if isinstance(fixture, Af1Simulator):
        valid_site_masks = ["funestus"]
    elif isinstance(fixture, Ag3Simulator):
        valid_site_masks = ["gamb_colu_arab", "gamb_colu", "arab"]
    else:
        valid_site_masks = [""] + list(api.site_mask_ids)
    site_mask = rng.choice(valid_site_masks)
    min_cohort_size = int(rng.integers(0, 2))
    transcript = random_transcript(api=api)
    cohorts = rng.choice(["admin1_year", "admin2_month", "country"])

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

    check_plot_frequencies_heatmap(api, df_snp_a)
    check_plot_frequencies_heatmap(api, df_snp_b)

    # Expect automatically deduplicate sample sets.
    assert_frame_equal(df_snp_b, df_snp_a)

    # Run the function under test.
    df_aa_a = api.aa_allele_frequencies(sample_sets=[sample_set], **params)
    df_aa_b = api.aa_allele_frequencies(sample_sets=[sample_set, sample_set], **params)

    # Expect automatically deduplicate sample sets.
    assert_frame_equal(df_aa_b, df_aa_a)


def check_snp_allele_frequencies_advanced(
    *,
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
    transcript=None,
    area_by="admin1_iso",
    period_by="year",
    sample_sets=None,
    sample_query=None,
    sample_query_options=None,
    min_cohort_size=None,
    nobs_mode="called",
    variant_query=None,
    site_mask=None,
):
    # Pick test parameters at random.
    if transcript is None:
        transcript = random_transcript(api=api).name
    if area_by is None:
        area_by = rng.choice(["country", "admin1_iso", "admin2_name"])
    if period_by is None:
        period_by = rng.choice(["year", "quarter", "month"])
    if sample_sets is None:
        all_sample_sets = api.sample_sets()["sample_set"].to_list()
        sample_sets = rng.choice(all_sample_sets)
    if min_cohort_size is None:
        min_cohort_size = int(rng.integers(0, 2))
    if site_mask is None:
        if isinstance(fixture, Af1Simulator):
            valid_site_masks = ["funestus"]
        elif isinstance(fixture, Ag3Simulator):
            valid_site_masks = ["gamb_colu_arab", "gamb_colu", "arab"]
        else:
            valid_site_masks = [""] + list(api.site_mask_ids)
        site_mask = rng.choice(valid_site_masks)

    # Run function under test.
    ds = api.snp_allele_frequencies_advanced(
        transcript=transcript,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        min_cohort_size=min_cohort_size,
        nobs_mode=nobs_mode,
        variant_query=variant_query,
        site_mask=site_mask,
    )

    # Check the result.
    assert isinstance(ds, xr.Dataset)
    check_plot_frequencies_time_series(api, ds)
    check_plot_frequencies_time_series_with_taxa(api, ds)
    check_plot_frequencies_time_series_with_areas(api, ds)
    check_plot_frequencies_interactive_map(api, ds)
    assert set(ds.dims) == {"cohorts", "variants"}

    # Check variant variables.
    expected_variant_vars = [
        "variant_label",
        "variant_contig",
        "variant_position",
        "variant_ref_allele",
        "variant_alt_allele",
        "variant_max_af",
        "variant_transcript",
        "variant_effect",
        "variant_impact",
        "variant_ref_codon",
        "variant_alt_codon",
        "variant_ref_aa",
        "variant_alt_aa",
        "variant_aa_pos",
        "variant_aa_change",
    ]
    expected_variant_vars += [f"variant_pass_{m}" for m in api.site_mask_ids]
    for v in expected_variant_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants",)

    # Check cohort variables.
    expected_cohort_vars = [
        "cohort_label",
        "cohort_size",
        "cohort_taxon",
        "cohort_area",
        "cohort_period",
        "cohort_period_start",
        "cohort_period_end",
        "cohort_lat_mean",
        "cohort_lat_min",
        "cohort_lat_max",
        "cohort_lon_mean",
        "cohort_lon_min",
        "cohort_lon_max",
    ]
    for v in expected_cohort_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("cohorts",)

    # Check event variables.
    expected_event_vars = [
        "event_count",
        "event_nobs",
        "event_frequency",
        "event_frequency_ci_low",
        "event_frequency_ci_upp",
    ]
    for v in expected_event_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants", "cohorts")

    # Sanity check for frequency values.
    x = ds["event_frequency"].values
    check_frequency(x)

    # Sanity check area values.
    df_samples = api.sample_metadata(
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
    )
    expected_area_values = np.unique(df_samples[area_by].dropna().values)
    area_values = ds["cohort_area"].values
    # N.B., some areas may not end up in final dataset if cohort
    # size is too small, so do a set membership test
    for a in area_values:
        assert a in expected_area_values

    # Sanity checks for period values.
    period_values = ds["cohort_period"].values
    if period_by == "year":
        expected_freqstr = "Y-DEC"
    elif period_by == "month":
        expected_freqstr = "M"
    elif period_by == "quarter":
        expected_freqstr = "Q-DEC"
    else:
        assert False, "not implemented"
    for p in period_values:
        assert isinstance(p, pd.Period)
        assert p.freqstr == expected_freqstr

    # Sanity check cohort sizes.
    cohort_size_values = ds["cohort_size"].values
    for s in cohort_size_values:
        assert s >= min_cohort_size

    if area_by == "admin1_iso" and period_by == "year" and nobs_mode == "called":
        # Here we test the behaviour of the function when grouping by admin level
        # 1 and year. We can do some more in-depth testing in this case because
        # we can compare results directly against the simpler snp_allele_frequencies()
        # function with the admin1_year cohorts.

        # Check consistency with the basic snp allele frequencies method.
        df_af = api.snp_allele_frequencies(
            transcript=transcript,
            cohorts="admin1_year",
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            min_cohort_size=min_cohort_size,
            site_mask=site_mask,
            include_counts=True,
        )
        # Make sure all variables available to check.
        df_af = df_af.reset_index()
        if variant_query is not None:
            df_af = df_af.query(variant_query)

        # Check cohorts are consistent.
        expect_cohort_labels = sorted(
            [c.split("frq_")[1] for c in df_af.columns if c.startswith("frq_")]
        )
        cohort_labels = sorted(ds["cohort_label"].values)
        assert cohort_labels == expect_cohort_labels

        # Check variants are consistent.
        assert ds.sizes["variants"] == len(df_af)
        for v in expected_variant_vars:
            c = v.split("variant_")[1]
            actual = ds[v]
            expect = df_af[c]
            compare_series_like(actual, expect)

        # Check frequencies are consistent.
        for cohort_index, cohort_label in enumerate(ds["cohort_label"].values):
            actual_nobs = ds["event_nobs"].values[:, cohort_index]
            expect_nobs = df_af[f"nobs_{cohort_label}"].values
            assert_array_equal(actual_nobs, expect_nobs)
            actual_count = ds["event_count"].values[:, cohort_index]
            expect_count = df_af[f"count_{cohort_label}"].values
            assert_array_equal(actual_count, expect_count)
            actual_frq = ds["event_frequency"].values[:, cohort_index]
            expect_frq = df_af[f"frq_{cohort_label}"].values
            assert_allclose(actual_frq, expect_frq)


def check_aa_allele_frequencies_advanced(
    *,
    api: AnophelesSnpFrequencyAnalysis,
    transcript=None,
    area_by="admin1_iso",
    period_by="year",
    sample_sets=None,
    sample_query=None,
    sample_query_options=None,
    min_cohort_size=None,
    nobs_mode="called",
    variant_query=None,
):
    # Pick test parameters at random.
    if transcript is None:
        transcript = random_transcript(api=api).name
    if area_by is None:
        area_by = rng.choice(["country", "admin1_iso", "admin2_name"])
    if period_by is None:
        period_by = rng.choice(["year", "quarter", "month"])
    if sample_sets is None:
        all_sample_sets = api.sample_sets()["sample_set"].to_list()
        sample_sets = rng.choice(all_sample_sets)
    if min_cohort_size is None:
        min_cohort_size = int(rng.integers(0, 2))

    # Run function under test.
    ds = api.aa_allele_frequencies_advanced(
        transcript=transcript,
        area_by=area_by,
        period_by=period_by,
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        min_cohort_size=min_cohort_size,
        nobs_mode=nobs_mode,
        variant_query=variant_query,
    )

    # Check the result.
    assert isinstance(ds, xr.Dataset)
    check_plot_frequencies_time_series(api, ds)
    check_plot_frequencies_time_series_with_taxa(api, ds)
    check_plot_frequencies_time_series_with_areas(api, ds)
    check_plot_frequencies_interactive_map(api, ds)
    assert set(ds.dims) == {"cohorts", "variants"}

    expected_variant_vars = (
        "variant_label",
        "variant_contig",
        "variant_position",
        "variant_max_af",
        "variant_transcript",
        "variant_effect",
        "variant_impact",
        "variant_ref_aa",
        "variant_alt_aa",
        "variant_aa_pos",
        "variant_aa_change",
    )
    for v in expected_variant_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants",)

    expected_cohort_vars = (
        "cohort_label",
        "cohort_size",
        "cohort_taxon",
        "cohort_area",
        "cohort_period",
        "cohort_period_start",
        "cohort_period_end",
        "cohort_lat_mean",
        "cohort_lat_min",
        "cohort_lat_max",
        "cohort_lon_mean",
        "cohort_lon_min",
        "cohort_lon_max",
    )
    for v in expected_cohort_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("cohorts",)

    expected_event_vars = (
        "event_count",
        "event_nobs",
        "event_frequency",
        "event_frequency_ci_low",
        "event_frequency_ci_upp",
    )
    for v in expected_event_vars:
        a = ds[v]
        assert isinstance(a, xr.DataArray)
        assert a.dims == ("variants", "cohorts")

    # Sanity check for frequency values.
    x = ds["event_frequency"].values
    check_frequency(x)

    # Sanity checks for area values.
    df_samples = api.sample_metadata(
        sample_sets=sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
    )
    expected_area_values = np.unique(df_samples[area_by].dropna().values)
    area_values = ds["cohort_area"].values
    # N.B., some areas may not end up in final dataset if cohort
    # size is too small, so do a set membership test
    for a in area_values:
        assert a in expected_area_values

    # Sanity checks for period values.
    period_values = ds["cohort_period"].values
    if period_by == "year":
        expected_freqstr = "Y-DEC"
    elif period_by == "month":
        expected_freqstr = "M"
    elif period_by == "quarter":
        expected_freqstr = "Q-DEC"
    else:
        assert False, "not implemented"
    for p in period_values:
        assert isinstance(p, pd.Period)
        assert p.freqstr == expected_freqstr

    # Sanity check cohort size.
    cohort_size_values = ds["cohort_size"].values
    for s in cohort_size_values:
        assert s >= min_cohort_size

    if area_by == "admin1_iso" and period_by == "year" and nobs_mode == "called":
        # Here we test the behaviour of the function when grouping by admin level
        # 1 and year. We can do some more in-depth testing in this case because
        # we can compare results directly against the simpler aa_allele_frequencies()
        # function with the admin1_year cohorts.

        # Check consistency with the basic aa allele frequencies method.
        df_af = api.aa_allele_frequencies(
            transcript=transcript,
            cohorts="admin1_year",
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            min_cohort_size=min_cohort_size,
            include_counts=True,
        )
        # Make sure all variables available to check.
        df_af = df_af.reset_index()
        if variant_query is not None:
            df_af = df_af.query(variant_query)

        # Check cohorts are consistent.
        expect_cohort_labels = sorted(
            [c.split("frq_")[1] for c in df_af.columns if c.startswith("frq_")]
        )
        cohort_labels = sorted(ds["cohort_label"].values)
        assert cohort_labels == expect_cohort_labels

        # Check variants are consistent.
        assert ds.sizes["variants"] == len(df_af)
        for v in expected_variant_vars:
            c = v.split("variant_")[1]
            actual = ds[v]
            expect = df_af[c]
            compare_series_like(actual, expect)

        # Check frequencies are consistent.
        for cohort_index, cohort_label in enumerate(ds["cohort_label"].values):
            actual_nobs = ds["event_nobs"].values[:, cohort_index]
            expect_nobs = df_af[f"nobs_{cohort_label}"].values
            assert_array_equal(actual_nobs, expect_nobs)
            actual_count = ds["event_count"].values[:, cohort_index]
            expect_count = df_af[f"count_{cohort_label}"].values
            assert_array_equal(actual_count, expect_count)
            actual_frq = ds["event_frequency"].values[:, cohort_index]
            expect_frq = df_af[f"frq_{cohort_label}"].values
            assert_allclose(actual_frq, expect_frq)


# Here we don't explore the full matrix, but vary one parameter at a time, otherwise
# the test suite would take too long to run.


@pytest.mark.parametrize("area_by", ["country", "admin1_iso", "admin2_name"])
@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_advanced_with_area_by(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
    area_by,
):
    check_snp_allele_frequencies_advanced(
        fixture=fixture,
        api=api,
        area_by=area_by,
    )
    check_aa_allele_frequencies_advanced(
        api=api,
        area_by=area_by,
    )


@pytest.mark.parametrize("period_by", ["year", "quarter", "month"])
@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_advanced_with_period_by(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
    period_by,
):
    check_snp_allele_frequencies_advanced(
        fixture=fixture,
        api=api,
        period_by=period_by,
    )
    check_aa_allele_frequencies_advanced(
        api=api,
        period_by=period_by,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_advanced_with_sample_query(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    df_samples = api.sample_metadata(sample_sets=all_sample_sets)
    countries = df_samples["country"].unique()
    country = rng.choice(countries)
    sample_query = f"country == '{country}'"

    check_snp_allele_frequencies_advanced(
        fixture=fixture,
        api=api,
        sample_sets=all_sample_sets,
        sample_query=sample_query,
        min_cohort_size=0,
    )
    check_aa_allele_frequencies_advanced(
        api=api,
        sample_sets=all_sample_sets,
        sample_query=sample_query,
        min_cohort_size=0,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_advanced_with_sample_query_options(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    df_samples = api.sample_metadata(sample_sets=all_sample_sets)
    countries = df_samples["country"].unique().tolist()
    countries_list = rng.choice(countries, 2, replace=False).tolist()
    sample_query_options = {
        "local_dict": {
            "countries_list": countries_list,
        }
    }
    sample_query = "country in @countries_list"

    check_snp_allele_frequencies_advanced(
        fixture=fixture,
        api=api,
        sample_sets=all_sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        min_cohort_size=0,
    )
    check_aa_allele_frequencies_advanced(
        api=api,
        sample_sets=all_sample_sets,
        sample_query=sample_query,
        sample_query_options=sample_query_options,
        min_cohort_size=0,
    )


@pytest.mark.parametrize("min_cohort_size", [0, 10, 100])
@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_advanced_with_min_cohort_size(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
    min_cohort_size,
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    area_by = "admin1_iso"
    period_by = "year"
    transcript = random_transcript(api=api).name

    if min_cohort_size <= 10:
        # Expect this to find at least one cohort, so go ahead with full
        # checks.
        check_snp_allele_frequencies_advanced(
            fixture=fixture,
            api=api,
            transcript=transcript,
            sample_sets=all_sample_sets,
            min_cohort_size=min_cohort_size,
            area_by=area_by,
            period_by=period_by,
        )
        check_aa_allele_frequencies_advanced(
            api=api,
            transcript=transcript,
            sample_sets=all_sample_sets,
            min_cohort_size=min_cohort_size,
            area_by=area_by,
            period_by=period_by,
        )
    else:
        # Expect this to find no cohorts.
        with pytest.raises(ValueError):
            api.snp_allele_frequencies_advanced(
                transcript=transcript,
                sample_sets=all_sample_sets,
                min_cohort_size=min_cohort_size,
                area_by=area_by,
                period_by=period_by,
            )
        with pytest.raises(ValueError):
            api.aa_allele_frequencies_advanced(
                transcript=transcript,
                sample_sets=all_sample_sets,
                min_cohort_size=min_cohort_size,
                area_by=area_by,
                period_by=period_by,
            )


@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_advanced_with_variant_query(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    area_by = "admin1_iso"
    period_by = "year"
    transcript = random_transcript(api=api).name

    # Test a query that should succeed.
    variant_query = "effect == 'NON_SYNONYMOUS_CODING'"
    check_snp_allele_frequencies_advanced(
        fixture=fixture,
        api=api,
        transcript=transcript,
        sample_sets=all_sample_sets,
        area_by=area_by,
        period_by=period_by,
        variant_query=variant_query,
    )
    check_aa_allele_frequencies_advanced(
        api=api,
        transcript=transcript,
        sample_sets=all_sample_sets,
        area_by=area_by,
        period_by=period_by,
        variant_query=variant_query,
    )

    # Test a query that should fail.
    variant_query = "effect == 'foobar'"
    with pytest.raises(ValueError):
        api.snp_allele_frequencies_advanced(
            transcript=transcript,
            sample_sets=all_sample_sets,
            area_by=area_by,
            period_by=period_by,
            variant_query=variant_query,
        )
    with pytest.raises(ValueError):
        api.aa_allele_frequencies_advanced(
            transcript=transcript,
            sample_sets=all_sample_sets,
            area_by=area_by,
            period_by=period_by,
            variant_query=variant_query,
        )


@pytest.mark.parametrize("nobs_mode", ["called", "fixed"])
@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_advanced_with_nobs_mode(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
    nobs_mode,
):
    check_snp_allele_frequencies_advanced(
        fixture=fixture,
        api=api,
        nobs_mode=nobs_mode,
    )
    check_aa_allele_frequencies_advanced(
        api=api,
        nobs_mode=nobs_mode,
    )


@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_advanced_with_dup_samples(
    fixture,
    api: AnophelesSnpFrequencyAnalysis,
):
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = rng.choice(all_sample_sets)
    sample_sets = [sample_set, sample_set]

    check_snp_allele_frequencies_advanced(
        fixture=fixture,
        api=api,
        sample_sets=sample_sets,
    )
    check_aa_allele_frequencies_advanced(
        api=api,
        sample_sets=sample_sets,
    )
