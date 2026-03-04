import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import pytest
from pytest_cases import parametrize_with_cases


from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data import adir1 as _adir1

from malariagen_data.anoph.pca import AnophelesPca
from malariagen_data.anoph import pca_params


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesPca(
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
    return AnophelesPca(
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
    return AnophelesPca(
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
def ag3_sim_api_local_path(ag3_sim_fixture):
    data_path = ag3_sim_fixture.bucket_path.as_posix()
    return AnophelesPca(
        url=data_path,
        public_url=data_path,
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


def case_adir1_sim(adir1_sim_fixture, adir1_sim_api):
    return adir1_sim_fixture, adir1_sim_api


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
    n_snps = random.randint(4, n_snps_available)
    # PC3 required for plot_pca_coords_3d()
    assert min(n_samples, n_snps) > 3
    n_components = random.randint(3, min(n_samples, n_snps, 10))

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


@parametrize_with_cases("fixture,api", cases=".")
def test_pca_exclude_samples(fixture, api: AnophelesPca):
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

    # Exclusion parameters.
    n_samples_excluded = random.randint(1, 5)
    samples = ds["sample_id"].values.tolist()
    exclude_samples = random.sample(samples, n_samples_excluded)

    # PCA parameters.
    n_samples = ds.sizes["samples"] - n_samples_excluded
    n_snps_available = ds.sizes["variants"]
    n_snps = random.randint(4, n_snps_available)
    n_components = random.randint(2, min(n_samples, n_snps, 10))

    # Run the PCA.
    pca_df, pca_evr = api.pca(
        n_snps=n_snps,
        n_components=n_components,
        exclude_samples=exclude_samples,
        **data_params,
    )

    # Check types.
    assert isinstance(pca_df, pd.DataFrame)
    assert isinstance(pca_evr, np.ndarray)

    # Check sizes.
    assert len(pca_df) == n_samples
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
    assert f"PC{n_components+1}" not in pca_df.columns
    assert "pca_fit" in pca_df.columns
    assert pca_df["pca_fit"].all()
    assert pca_evr.ndim == 1
    assert pca_evr.shape[0] == n_components

    # Check exclusions.
    assert len(pca_df.query(f"sample_id in {exclude_samples}")) == 0


@parametrize_with_cases("fixture,api", cases=".")
def test_pca_fit_exclude_samples(fixture, api: AnophelesPca):
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

    # Exclusion parameters.
    n_samples_excluded = random.randint(1, 5)
    samples = ds["sample_id"].values.tolist()
    exclude_samples = random.sample(samples, n_samples_excluded)

    # PCA parameters.
    n_samples = ds.sizes["samples"]
    n_snps_available = ds.sizes["variants"]
    n_snps = random.randint(4, n_snps_available)
    n_components = random.randint(2, min(n_samples, n_snps, 10))

    # Run the PCA.
    pca_df, pca_evr = api.pca(
        n_snps=n_snps,
        n_components=n_components,
        fit_exclude_samples=exclude_samples,
        **data_params,
    )

    # Check types.
    assert isinstance(pca_df, pd.DataFrame)
    assert isinstance(pca_evr, np.ndarray)

    # Check sizes.
    assert len(pca_df) == n_samples
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
    assert f"PC{n_components+1}" not in pca_df.columns
    assert "pca_fit" in pca_df.columns
    assert pca_evr.ndim == 1
    assert pca_evr.shape[0] == n_components

    # Check exclusions.
    assert not pca_df["pca_fit"].all()
    assert pca_df["pca_fit"].sum() == n_samples - n_samples_excluded
    assert len(pca_df.query(f"sample_id in {exclude_samples}")) == n_samples_excluded
    assert (
        len(pca_df.query(f"sample_id in {exclude_samples} and not pca_fit"))
        == n_samples_excluded
    )


def test_pca_cohort_size_column_requires_cohort_size(ag3_sim_api_local_path):
    api = ag3_sim_api_local_path
    sample_set = api.sample_sets()["sample_set"].iloc[0]
    with pytest.raises(ValueError, match="cohort_size must be provided"):
        api.pca(
            region=api.contigs[0],
            n_snps=4,
            n_components=2,
            sample_sets=[sample_set],
            cohort_size_column="country",
        )


def test_pca_cohort_size_column_downsamples_per_cohort(ag3_sim_api_local_path):
    api = ag3_sim_api_local_path
    df_samples = api.sample_metadata()
    cohort_size = 2
    cohort_counts = df_samples["country"].value_counts(dropna=True)
    eligible_cohorts = cohort_counts[cohort_counts >= cohort_size]
    if len(eligible_cohorts) < 2:
        pytest.skip("not enough simulated cohorts with sufficient sample size")
    selected_cohorts = eligible_cohorts.index[:2].to_list()

    pca_df, _ = api.pca(
        region=api.contigs[0],
        n_snps=4,
        n_components=2,
        cohort_size=cohort_size,
        cohort_size_column="country",
        sample_query=f"country in {selected_cohorts!r}",
        random_seed=42,
    )

    selected_counts = pca_df["country"].value_counts()
    assert set(selected_counts.index.to_list()) == set(selected_cohorts)
    assert (selected_counts == cohort_size).all()


def test_pca_cohort_size_column_updates_sample_sets_after_downsampling(
    ag3_sim_api_local_path, monkeypatch
):
    api = ag3_sim_api_local_path
    all_sample_sets = api.sample_sets()["sample_set"].to_list()
    if len(all_sample_sets) < 2:
        pytest.skip("not enough simulated sample sets")
    dropped_sample_set = all_sample_sets[-1]
    cohort_size = 2

    original_sample_metadata = api.sample_metadata

    def sample_metadata_with_missing_cohort_column(*args, **kwargs):
        df = original_sample_metadata(*args, **kwargs)
        df = df.copy()
        df.loc[df["sample_set"] == dropped_sample_set, "country"] = np.nan
        return df

    monkeypatch.setattr(
        api, "sample_metadata", sample_metadata_with_missing_cohort_column
    )

    (
        prepared_sample_sets,
        prepared_sample_indices,
    ) = api._prep_sample_selection_cache_params(
        sample_sets=all_sample_sets,
        sample_query=None,
        sample_query_options=None,
        sample_indices=None,
    )
    (
        updated_sample_sets,
        updated_sample_indices,
    ) = api._downsample_sample_indices_by_cohort(
        sample_sets=prepared_sample_sets,
        sample_indices=prepared_sample_indices,
        cohort_size=cohort_size,
        cohort_size_column="country",
        random_seed=42,
    )

    assert dropped_sample_set not in updated_sample_sets
    assert updated_sample_indices
    selected_df = api.sample_metadata(sample_sets=updated_sample_sets).iloc[
        updated_sample_indices
    ]
    assert dropped_sample_set not in selected_df["sample_set"].to_list()


def test_pca_cohort_size_column_skips_small_cohorts_with_warning(
    ag3_sim_api_local_path, monkeypatch
):
    api = ag3_sim_api_local_path
    cohort_size = 2
    original_sample_metadata = api.sample_metadata

    def sample_metadata_with_small_cohort(*args, **kwargs):
        df = original_sample_metadata(*args, **kwargs)
        df = df.copy()
        if not df.empty:
            df.loc[df.index[0], "country"] = "__tiny_cohort__"
        return df

    monkeypatch.setattr(api, "sample_metadata", sample_metadata_with_small_cohort)

    with pytest.warns(UserWarning, match="Skipping cohort"):
        pca_df, _ = api.pca(
            region=api.contigs[0],
            n_snps=4,
            n_components=2,
            cohort_size=cohort_size,
            cohort_size_column="country",
            random_seed=42,
        )

    assert "__tiny_cohort__" not in pca_df["country"].to_list()
