import random

import bokeh.models
import pandas as pd
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data import adir1 as _adir1
from malariagen_data import amin1 as _amin1
from malariagen_data.anoph.heterozygosity import AnophelesHetAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesHetAnalysis(
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
        virtual_contigs=_ag3.VIRTUAL_CONTIGS,
    )


@pytest.fixture
def af1_sim_api(af1_sim_fixture):
    return AnophelesHetAnalysis(
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


@pytest.fixture
def adir1_sim_api(adir1_sim_fixture):
    return AnophelesHetAnalysis(
        url=adir1_sim_fixture.url,
        public_url=adir1_sim_fixture.url,
        config_path=_adir1.CONFIG_PATH,
        major_version_number=_adir1.MAJOR_VERSION_NUMBER,
        major_version_path=_adir1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        default_site_mask="dirus",
        results_cache=adir1_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_adir1.TAXON_COLORS,
    )


@pytest.fixture
def amin1_sim_api(amin1_sim_fixture):
    return AnophelesHetAnalysis(
        url=amin1_sim_fixture.url,
        public_url=amin1_sim_fixture.url,
        config_path=_amin1.CONFIG_PATH,
        major_version_number=_amin1.MAJOR_VERSION_NUMBER,
        major_version_path=_amin1.MAJOR_VERSION_PATH,
        pre=False,
        gff_gene_type="protein_coding_gene",
        gff_gene_name_attribute="Note",
        gff_default_attributes=("ID", "Parent", "Note", "description"),
        default_site_mask="minimus",
        results_cache=amin1_sim_fixture.results_cache_path.as_posix(),
        taxon_colors=_amin1.TAXON_COLORS,
    )


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


def case_adir1_sim(adir1_sim_fixture, adir1_sim_api):
    return adir1_sim_fixture, adir1_sim_api


def case_amin1_sim(amin1_sim_fixture, amin1_sim_api):
    return amin1_sim_fixture, amin1_sim_api


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_heterozygosity_track(fixture, api: AnophelesHetAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = random.choice(all_sample_sets)
    df_samples = api.sample_metadata(sample_sets=sample_set)
    sample = random.choice(df_samples["sample_id"].to_list())

    het_params = dict(
        sample=sample,
        region=random.choice(api.contigs),
        sample_set=sample_set,
        window_size=20_000,
    )

    # Run function under test.
    fig = api.plot_heterozygosity_track(**het_params, show=False)

    # Check results.
    assert isinstance(fig, bokeh.models.Plot)


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_heterozygosity(fixture, api: AnophelesHetAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = random.choice(all_sample_sets)
    df_samples = api.sample_metadata(sample_sets=sample_set)
    sample = random.choice(df_samples["sample_id"].to_list())

    het_params = dict(
        sample=sample,
        region=random.choice(api.contigs),
        sample_set=sample_set,
        window_size=20_000,
    )

    # Run function under test.
    fig = api.plot_heterozygosity(**het_params, show=False)

    # Check results.
    assert isinstance(fig, bokeh.models.GridPlot)


@parametrize_with_cases("fixture,api", cases=".")
def test_roh_hmm(fixture, api: AnophelesHetAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = random.choice(all_sample_sets)
    df_samples = api.sample_metadata(sample_sets=sample_set)
    sample = random.choice(df_samples["sample_id"].to_list())

    roh_params = dict(
        sample=sample,
        region=random.choice(api.contigs),
        sample_set=sample_set,
        window_size=20_000,
    )

    # Run function under test.
    df_roh = api.roh_hmm(**roh_params)

    # Check results.
    assert isinstance(df_roh, pd.DataFrame)
    expected_columns = [
        "sample_id",
        "contig",
        "roh_start",
        "roh_stop",
        "roh_length",
        "roh_is_marginal",
    ]
    for col in expected_columns:
        assert col in df_roh.columns


@parametrize_with_cases("fixture,api", cases=".")
def test_plot_roh(fixture, api: AnophelesHetAnalysis):
    # Set up test parameters.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    sample_set = random.choice(all_sample_sets)
    df_samples = api.sample_metadata(sample_sets=sample_set)
    sample = random.choice(df_samples["sample_id"].to_list())

    roh_params = dict(
        sample=sample,
        region=random.choice(api.contigs),
        sample_set=sample_set,
        window_size=20_000,
    )

    # Run function under test.
    fig = api.plot_roh(**roh_params, show=False)

    # Check results.
    assert isinstance(fig, bokeh.models.GridPlot)
