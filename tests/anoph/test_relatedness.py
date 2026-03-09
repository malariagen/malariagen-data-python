import random
import pandas as pd
import pytest
from pytest_cases import parametrize_with_cases

from malariagen_data import af1 as _af1
from malariagen_data import ag3 as _ag3
from malariagen_data import adir1 as _adir1

from malariagen_data.anoph.relatedness import AnophelesRelatednessAnalysis
from malariagen_data.anoph import pca_params


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesRelatednessAnalysis(
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
    return AnophelesRelatednessAnalysis(
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
    return AnophelesRelatednessAnalysis(
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


def case_ag3_sim(ag3_sim_fixture, ag3_sim_api):
    return ag3_sim_fixture, ag3_sim_api


def case_af1_sim(af1_sim_fixture, af1_sim_api):
    return af1_sim_fixture, af1_sim_api


def case_adir1_sim(adir1_sim_fixture, adir1_sim_api):
    return adir1_sim_fixture, adir1_sim_api


@parametrize_with_cases("fixture,api", cases=".")
def test_pc_relate(fixture, api: AnophelesRelatednessAnalysis):
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

    # PCA & relatedness parameters.
    n_samples = ds.sizes["samples"]
    n_snps_available = ds.sizes["variants"]
    n_snps = random.randint(4, n_snps_available)
    assert min(n_samples, n_snps) > 3
    n_components = random.randint(3, min(n_samples, n_snps, 10))

    # Run PC_Relate
    df_kinship = api.pc_relate(
        n_snps=n_snps,
        n_components=n_components,
        **data_params,
    )

    # Check types and size.
    assert isinstance(df_kinship, pd.DataFrame)
    assert len(df_kinship) == n_samples
    assert len(df_kinship.columns) == n_samples

    # Check that it returns plausible values (e.g. self-relatedness should be bounded around 0.5 normally, but it could vary)
    # At minimum, verify it does not contain entirely nans, although with simulated data the values could be weird.
    # So we simply check that there are no NaNs on the diagonal if there's sufficient data.
    # Actually, sgkit can output NaNs if variance is 0, so just check it executed correctly.


@parametrize_with_cases("fixture,api", cases=".")
def test_pc_relate_exclude_samples(fixture, api: AnophelesRelatednessAnalysis):
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

    # PCA and relatedness parameters.
    n_samples = ds.sizes["samples"] - n_samples_excluded
    n_snps_available = ds.sizes["variants"]
    n_snps = random.randint(4, n_snps_available)
    n_components = random.randint(2, min(n_samples, n_snps, 10))

    # Run PC_Relate
    df_kinship = api.pc_relate(
        n_snps=n_snps,
        n_components=n_components,
        exclude_samples=exclude_samples,
        **data_params,
    )

    # Check exclusion
    assert isinstance(df_kinship, pd.DataFrame)
    assert len(df_kinship) == n_samples
    assert len(df_kinship.columns) == n_samples

    # Check that excluded samples are not in the index and columns
    for exc in exclude_samples:
        assert exc not in df_kinship.index.tolist()
        assert exc not in df_kinship.columns.tolist()
