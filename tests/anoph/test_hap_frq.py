import random

import pandas as pd
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.hap_freq import AnophelesHapFrequencyAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesHapFrequencyAnalysis(
        url=ag3_sim_fixture.url,
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
        taxon_colors=_ag3.TAXON_COLORS,
        default_phasing_analysis="gamb_colu_arab",
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
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


def check_hap_frequencies(
    *,
    api,
    df,
):
    assert isinstance(df, pd.DataFrame)


@pytest.mark.parametrize(
    "cohorts", ["admin1_year", "admin2_month", "country", "foobar"]
)
@parametrize_with_cases("fixture,api", cases=".")
def test_allele_frequencies_with_str_cohorts(
    fixture,
    api: AnophelesHapFrequencyAnalysis,
    cohorts,
):
    # Pick test parameters at random.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_sets = random.choice(all_sample_sets)
    min_cohort_size = random.randint(0, 2)
    region = fixture.random_region_str()

    # Set up call params.
    params = dict(
        region=region,
        cohorts=cohorts,
        min_cohort_size=min_cohort_size,
        sample_sets=sample_sets,
    )

    # Test behaviour with bad cohorts param.
    if cohorts == "foobar":
        with pytest.raises(ValueError):
            api.haplotype_frequencies(**params)
        return

    # Run the function under test.
    df_snp = api.haplotype_frequencies(**params)

    # Standard checks.
    check_hap_frequencies(api=api, df=df_snp)
