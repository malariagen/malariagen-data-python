# import itertools
import random
import pytest
from pytest_cases import parametrize_with_cases
import numpy as np
# import bokeh.models
# import pandas as pd
# import plotly.graph_objects as go

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.admixture_stats import AnophelesAdmixtureAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesAdmixtureAnalysis(
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
        default_site_mask="gamb_colu_arab",
        results_cache=ag3_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_ag3.TAXON_COLORS,
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesAdmixtureAnalysis(
        url=af1_sim_fixture.url,
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


@parametrize_with_cases("fixture,api", cases=".")
def test_patterson_f3(fixture, api: AnophelesAdmixtureAnalysis):
    # Set up test parameters.
    all_taxon = api.sample_metadata()["taxon"].to_list()
    taxon = random.sample(all_taxon, 3)
    recipient_query = f"taxon == {taxon[0]!r}"
    source1_query = f"taxon == {taxon[1]!r}"
    source2_query = f"taxon == {taxon[2]!r}"
    admixture_params = dict(
        region=random.choice(api.contigs),
        recipient_query=recipient_query,
        source1_query=source1_query,
        source2_query=source2_query,
        site_mask=random.choice(api.site_mask_ids),
        segregating_mode=random.choice(["recipient", "all"]),
        n_jack=random.randint(10, 200),
        min_cohort_size=1,
        max_cohort_size=50,
    )

    # Run patterson f3 function under test.
    f3, se, z = api.patterson_f3(**admixture_params)

    # Check results.
    assert isinstance(f3, np.float64)
    assert isinstance(se, np.float64)
    assert isinstance(z, np.float64)
    assert -0.1 <= f3 <= 1
    assert 0 <= se <= 1
    # assert -50 <= z <= 50 The sample is taken from across all data and it seems this value can be very large.
