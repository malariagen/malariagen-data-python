import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.pca import AnophelesPca
from malariagen_data.anoph import pca_params


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesPca(
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
    return AnophelesPca(
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
def test_pca_plotting(fixture, api: AnophelesPca):
    # Parameters for selecting input data.
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    data_params = dict(
        region=random.choice(api.contigs),
        sample_sets=random.sample(all_sample_sets, 2),
        site_mask=random.choice((None,) + api.site_mask_ids),
    )
    ds = api.biallelic_snp_calls(
        min_minor_ac=pca_params.min_minor_ac_default,
        max_missing_an=pca_params.max_missing_an_default,
        **data_params,
    )

    # PCA parameters.
    n_samples = ds.sizes["samples"]
    n_snps_available = ds.sizes["variants"]
    n_snps = random.randint(1, n_snps_available)
    # PC3 required for plot_pca_coords_3d()
    n_components = random.randint(3, min(n_samples, n_snps))

    # Run the PCA.
    pca_df, pca_evr = api.pca(
        n_snps=n_snps,
        n_components=n_components,
        **data_params,
    )

    # Check types.
    assert isinstance(pca_df, pd.DataFrame)
    assert isinstance(pca_evr, np.ndarray)

    # Check sizes.
    assert len(pca_df) == ds.sizes["samples"]
    for i in range(n_components):
        assert f"PC{i+1}" in pca_df.columns, (
            "n_components",
            n_components,
            "n_samples",
            n_samples,
            "n_snps_available",
            n_snps_available,
            "n_snps",
            n_snps,
        )
    assert pca_evr.ndim == 1
    assert pca_evr.shape[0] == n_components

    # Plot variance explained.
    fig_evr = api.plot_pca_variance(
        evr=pca_evr,
        show=False,
    )
    assert isinstance(fig_evr, go.Figure)

    # Set up parametrization of plotting parameters.
    custom_cohorts = {
        "male": "sex_call == 'M'",
        "female": "sex_call == 'F'",
    }
    colors = [None, "taxon", "country", "admin1_year", custom_cohorts]
    symbols = ["country", None, custom_cohorts, "admin2_month", "taxon"]

    # Test plotting with some different parameter combinations.
    for color, symbol in zip(colors, symbols):
        # Plot 2D coords.
        fig_2d = api.plot_pca_coords(
            data=pca_df,
            show=False,
            color=color,
            symbol=symbol,
        )
        assert isinstance(fig_2d, go.Figure)

        # Plot 2D coords.
        fig_3d = api.plot_pca_coords_3d(
            data=pca_df,
            show=False,
            color=color,
            symbol=symbol,
        )
        assert isinstance(fig_3d, go.Figure)
