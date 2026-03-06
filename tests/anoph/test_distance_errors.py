import pytest
from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.distance import AnophelesDistanceAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    return AnophelesDistanceAnalysis(
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


def test_plot_njt_no_samples(ag3_sim_api):
    # Test with a query matching no samples.
    with pytest.raises(ValueError) as e:
        ag3_sim_api.plot_njt(
            region="2L", n_snps=10, sample_query="sex_call == 'Impossible_Value'"
        )
    assert "No samples found for query" in str(
        e.value
    ) or "No relevant samples found" in str(e.value)


def test_plot_njt_not_enough_snps(ag3_sim_api):
    # Request more SNPs than available in the region
    with pytest.raises(ValueError) as e:
        ag3_sim_api.plot_njt(region="2L", n_snps=10000000, sample_query=None)
    assert "Not enough SNPs." in str(e.value)
    assert "Found" in str(e.value)
    assert "needed 10000000" in str(e.value)


def test_plot_njt_one_sample(ag3_sim_api):
    # Test with a query that returns only 1 sample.
    # This should trigger the minimum sample check in plot_njt.

    # First, find a sample so we can query for just one
    df_samples = ag3_sim_api.sample_metadata()
    sample_id = df_samples.iloc[0]["sample_id"]

    with pytest.raises(ValueError) as e:
        ag3_sim_api.plot_njt(
            region="2L", n_snps=10, sample_query=f"sample_id == '{sample_id}'"
        )
    assert "Not enough samples for neighbour-joining tree" in str(e.value)
    assert "Found 1" in str(e.value)
    assert "needed at least 2" in str(e.value)
