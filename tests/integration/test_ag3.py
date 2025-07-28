import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from malariagen_data import Ag3, Region
from malariagen_data.util import locate_region, resolve_region
import xarray as xr


contigs = "2R", "2L", "3R", "3L", "X"


def setup_ag3(url="simplecache::gs://vo_agam_release_master_us_central1/", **kwargs):
    kwargs.setdefault("check_location", False)
    kwargs.setdefault("show_progress", False)
    if url is None:
        # test default URL
        # This only tests the setup_af1 default url, not the Ag3 default.
        # The test_anopheles setup_subclass tests true defaults.
        return Ag3(**kwargs)
    if url.startswith("simplecache::"):
        # configure the directory on the local file system to cache data
        kwargs["simplecache"] = dict(cache_storage="gcs_cache")
    return Ag3(url, **kwargs)


def test_repr():
    ag3 = setup_ag3(check_location=True)
    assert isinstance(ag3, Ag3)
    r = repr(ag3)
    assert isinstance(r, str)


def test_cross_metadata():
    ag3 = setup_ag3()
    df_crosses = ag3.cross_metadata()
    assert isinstance(df_crosses, pd.DataFrame)
    expected_cols = ["cross", "sample_id", "father_id", "mother_id", "sex", "role"]
    assert df_crosses.columns.tolist() == expected_cols

    # check samples are in AG1000G-X
    df_samples = ag3.sample_metadata(sample_sets="AG1000G-X")
    assert set(df_crosses["sample_id"]) == set(df_samples["sample_id"])

    # check values
    expected_role_values = ["parent", "progeny"]
    assert df_crosses["role"].unique().tolist() == expected_role_values
    expected_sex_values = ["F", "M"]
    assert df_crosses["sex"].unique().tolist() == expected_sex_values


@pytest.mark.parametrize(
    "region_raw",
    [
        "AGAP007280",
        "3L",
        "2R:48714463-48715355",
        "2R:24,630,355-24,633,221",
        Region("2R", 48714463, 48715355),
    ],
)
def test_locate_region(region_raw):
    # TODO Migrate this test.
    ag3 = setup_ag3()
    gene_annotation = ag3.genome_features(attributes=["ID"])
    region = resolve_region(ag3, region_raw)
    pos = ag3.snp_sites(region=region.contig, field="POS")
    ref = ag3.snp_sites(region=region.contig, field="REF")
    loc_region = locate_region(region, pos)

    # check types
    assert isinstance(loc_region, slice)
    assert isinstance(region, Region)

    # check Region with contig
    if region_raw == "3L":
        assert region.contig == "3L"
        assert region.start is None
        assert region.end is None

    # check that Region goes through unchanged
    if isinstance(region_raw, Region):
        assert region == region_raw

    # check that gene name matches coordinates from the genome_features and matches gene sequence
    if region_raw == "AGAP007280":
        gene = gene_annotation.query("ID == 'AGAP007280'").squeeze()
        assert region == Region(gene.contig, gene.start, gene.end)
        assert pos[loc_region][0] == gene.start
        assert pos[loc_region][-1] == gene.end
        assert (
            ref[loc_region][:5].compute()
            == np.array(["A", "T", "G", "G", "C"], dtype="S1")
        ).all()

    # check string parsing
    if region_raw == "2R:48714463-48715355":
        assert region == Region("2R", 48714463, 48715355)
    if region_raw == "2R:24,630,355-24,633,221":
        assert region == Region("2R", 24630355, 24633221)


def test_ihs_gwss():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    sample_query = "country == 'Ghana'"
    contig = "3L"
    analysis = "gamb_colu"
    sample_sets = "3.0"
    window_size = 1000

    x, ihs = ag3.ihs_gwss(
        contig=contig,
        analysis=analysis,
        sample_query=sample_query,
        sample_sets=sample_sets,
        window_size=window_size,
        max_cohort_size=20,
    )

    assert isinstance(x, np.ndarray)
    assert isinstance(ihs, np.ndarray)

    # check dimensions
    assert len(x) == 395
    assert len(x) == len(ihs)

    # check some values
    assert_allclose(x[0], 510232.847)
    assert_allclose(ihs[:, 2][100], 2.3467595962486327)


def test_xpehh_gwss():
    ag3 = setup_ag3(cohorts_analysis="20230516")
    cohort1_query = "country == 'Ghana'"
    cohort2_query = "country == 'Angola'"
    contig = "3L"
    analysis = "gamb_colu"
    sample_sets = "3.0"
    window_size = 1000

    x, xpehh = ag3.xpehh_gwss(
        contig=contig,
        analysis=analysis,
        cohort1_query=cohort1_query,
        cohort2_query=cohort2_query,
        sample_sets=sample_sets,
        window_size=window_size,
        max_cohort_size=20,
    )

    assert isinstance(x, np.ndarray)
    assert isinstance(xpehh, np.ndarray)

    # check dimensions
    assert len(x) == 399
    assert len(x) == len(xpehh)

    # check some values
    assert_allclose(x[0], 467448.348)
    assert_allclose(xpehh[:, 2][100], 0.4817561326426265)


@pytest.mark.parametrize(
    "inversion",
    ["2La", "2Rb", "2Rc_col", "X_x"],
)
def test_karyotyping(inversion):
    ag3 = setup_ag3(cohorts_analysis="20230516")

    if inversion == "X_x":
        with pytest.raises(ValueError):
            ag3.karyotype(
                inversion=inversion, sample_sets="AG1000G-GH", sample_query=None
            )
    else:
        df = ag3.karyotype(
            inversion=inversion, sample_sets="AG1000G-GH", sample_query=None
        )
        assert isinstance(df, pd.DataFrame)
        expected_cols = [
            "sample_id",
            "inversion",
            f"karyotype_{inversion}_mean",
            f"karyotype_{inversion}",
            "total_tag_snps",
        ]
        assert set(df.columns) == set(expected_cols)
        assert all(df[f"karyotype_{inversion}"].isin([0, 1, 2]))
        assert all(df[f"karyotype_{inversion}_mean"].between(0, 2))

def test_plot_haplotype_network_string_direct(mocker):
    ag3 = setup_ag3()
    mocker.patch("dash.Dash.run")
    mock_mjn = mocker.patch("malariagen_data.anopheles.mjn_graph")
    mock_mjn.return_value = ([{"data": {"id": "n1"}}], [])

    ag3.plot_haplotype_network(
        region="2L:2,358,158-2,358,258",
        analysis="gamb_colu",
        sample_sets="3.0",
        sample_query="taxon == 'coluzzii'",
        color="country",
        max_dist=2,
        server_mode="inline",
    )

    assert mock_mjn.called
    call_args = mock_mjn.call_args[1]
    assert call_args["color"] == "partition"
    assert call_args["ht_color_counts"] is not None


def test_plot_haplotype_network_string_cohort(mocker):
    ag3 = setup_ag3()
    mocker.patch("dash.Dash.run")
    mock_mjn = mocker.patch("malariagen_data.anopheles.mjn_graph")
    mock_mjn.return_value = ([{"data": {"id": "n1"}}], [])

    ag3.plot_haplotype_network(
        region="2L:2,358,158-2,358,258",
        analysis="gamb_colu",
        sample_sets="3.0",
        sample_query="taxon == 'coluzzii'",
        color="admin1_iso",
        max_dist=2,
        server_mode="inline",
    )

    assert mock_mjn.called
    call_args = mock_mjn.call_args[1]
    assert call_args["color"] == "partition"
    assert call_args["ht_color_counts"] is not None


def test_plot_haplotype_network_mapping(mocker):
    ag3 = setup_ag3()
    mocker.patch("dash.Dash.run")
    mock_mjn = mocker.patch("malariagen_data.anopheles.mjn_graph")
    mock_mjn.return_value = ([{"data": {"id": "n1"}}], [])

    color_mapping = {"Ghana": "country == 'Ghana'", "Other": "country != 'Ghana'"}
    ag3.plot_haplotype_network(
        region="2L:2,358,158-2,358,258",
        analysis="gamb_colu",
        sample_sets="3.0",
        sample_query="taxon == 'coluzzii'",
        color=color_mapping,
        max_dist=2,
        server_mode="inline",
    )

    assert mock_mjn.called
    call_args = mock_mjn.call_args[1]
    assert call_args["color"] == "partition"
    assert call_args["ht_color_counts"] is not None


def test_plot_haplotype_network_none(mocker):
    ag3 = setup_ag3()
    mocker.patch("dash.Dash.run")
    mock_mjn = mocker.patch("malariagen_data.anopheles.mjn_graph")
    mock_mjn.return_value = ([{"data": {"id": "n1"}}], [])

    ag3.plot_haplotype_network(
        region="2L:2,358,158-2,358,258",
        analysis="gamb_colu",
        sample_sets="3.0",
        sample_query="taxon == 'coluzzii'",
        color=None,
        max_dist=2,
        server_mode="inline",
    )

    assert mock_mjn.called
    call_args = mock_mjn.call_args[1]
    assert call_args["color"] is None
    assert call_args["ht_color_counts"] is None

def test_phenotype_data():
    """Test basic functionality of phenotype_data method with sample_query."""
    ag3 = setup_ag3()

    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"
    insecticide_query = "Deltamethrin"

    # Use sample_query for filtering
    df = ag3.phenotype_data(
        sample_sets=sample_set, sample_query=f"insecticide == '{insecticide_query}'"
    )

    assert isinstance(df, pd.DataFrame)

    expected_cols = ["sample_id", "insecticide", "dose", "phenotype", "sample_set"]
    for col in expected_cols:
        assert col in df.columns

    # Check data content
    assert len(df) > 0
    assert all(df["insecticide"] == insecticide_query)  # Check if filter was applied
    assert set(df["phenotype"].str.lower().unique()).issubset(
        {
            "alive",
            "dead",
            "resistant",
            "susceptible",
            "died",
            "survived",
        }  # Include all possible values
    )
    assert sample_set in df["sample_set"].unique()

    # Test with multiple query conditions
    df_multi_query = ag3.phenotype_data(
        sample_sets=sample_set,
        sample_query=f"insecticide == '{insecticide_query}' and dose == 0.5",
    )
    assert isinstance(df_multi_query, pd.DataFrame)
    assert len(df_multi_query) > 0
    assert all(df_multi_query["insecticide"] == insecticide_query)
    assert all(df_multi_query["dose"] == 0.5)


def test_phenotype_binary_functionality():
    """Test phenotype_binary method using sample_query for filtering."""
    ag3 = setup_ag3()

    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"  # Use a known sample set for testing

    # Test 1: Filter by insecticide
    query_insecticide = "insecticide == 'Deltamethrin'"
    binary_series_insecticide = ag3.phenotype_binary(
        sample_sets=[sample_set], sample_query=query_insecticide
    )
    assert isinstance(binary_series_insecticide, pd.Series)
    assert binary_series_insecticide.name == "phenotype_binary"
    assert len(binary_series_insecticide) > 0
    assert set(binary_series_insecticide.unique()).issubset({0.0, 1.0, np.nan})

    # Test 2: Filter by multiple conditions (insecticide and phenotype outcome)
    query_multi = "insecticide == 'Deltamethrin' and phenotype == 'alive'"
    binary_series_multi = ag3.phenotype_binary(
        sample_sets=[sample_set], sample_query=query_multi
    )
    assert isinstance(binary_series_multi, pd.Series)
    assert binary_series_multi.name == "phenotype_binary"
    assert len(binary_series_multi) > 0
    assert set(binary_series_multi.unique()).issubset(
        {1.0, np.nan}
    )  # Expect mostly 1.0s

    # Test 3: Filter by dose
    query_dose = "dose == 0.5"
    binary_series_dose = ag3.phenotype_binary(
        sample_sets=[sample_set], sample_query=query_dose
    )
    assert isinstance(binary_series_dose, pd.Series)
    assert len(binary_series_dose) > 0
    assert set(binary_series_dose.unique()).issubset(
        {0.0, 1.0, np.nan}
    )  # Depending on data, could be 0.0, 1.0 or both

    # Test 4: Test with no matching data (should return empty Series)
    query_no_match = "insecticide == 'NonExistentInsecticide'"
    binary_series_empty = ag3.phenotype_binary(
        sample_sets=[sample_set], sample_query=query_no_match
    )
    assert isinstance(binary_series_empty, pd.Series)
    assert binary_series_empty.empty
    assert binary_series_empty.name == "phenotype_binary"


@pytest.mark.parametrize(
    "cohort_param,expected_result",
    [
        ({"min_cohort_size": 5}, "min_size"),
        ({"max_cohort_size": 10}, "max_size"),
        ({"cohort_size": 8}, "exact_size"),
    ],
)
def test_cohort_filtering(cohort_param, expected_result):
    """Test cohort size filtering functionality with sample_query."""
    ag3 = setup_ag3()

    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"
    insecticide_query = "Deltamethrin"

    # Apply cohort filtering with sample_query
    df_filtered = ag3.phenotype_data(
        sample_sets=sample_set,
        sample_query=f"insecticide == '{insecticide_query}'",
        **cohort_param,
    )

    assert isinstance(df_filtered, pd.DataFrame)

    if expected_result == "min_size":
        cohort_keys = ["insecticide", "dose", "location", "country", "sample_set"]
        available_keys = [col for col in cohort_keys if col in df_filtered.columns]
        if available_keys and not df_filtered.empty:
            cohort_sizes = df_filtered.groupby(available_keys).size()
            assert all(size >= cohort_param["min_cohort_size"] for size in cohort_sizes)
        elif df_filtered.empty and cohort_param["min_cohort_size"] > 0:
            # If no data meets criteria, ensure it's empty
            pass

    elif expected_result == "max_size":
        cohort_keys = ["insecticide", "dose", "location", "country", "sample_set"]
        available_keys = [col for col in cohort_keys if col in df_filtered.columns]
        if available_keys and not df_filtered.empty:
            cohort_sizes = df_filtered.groupby(available_keys).size()
            assert all(size <= cohort_param["max_cohort_size"] for size in cohort_sizes)

    elif expected_result == "exact_size":
        cohort_keys = ["insecticide", "dose", "location", "country", "sample_set"]
        available_keys = [col for col in cohort_keys if col in df_filtered.columns]
        if available_keys and not df_filtered.empty:
            cohort_sizes = df_filtered.groupby(available_keys).size()
            assert all(size == cohort_param["cohort_size"] for size in cohort_sizes)
        elif df_filtered.empty and cohort_param["cohort_size"] > 0:
            pass  # Acceptable if no cohorts meet exact size


def test_sample_query_functionality():
    """Test sample_query functionality."""
    ag3 = setup_ag3()

    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"

    df_baseline = ag3.phenotype_data(sample_sets=sample_set)

    if not df_baseline.empty and "location" in df_baseline.columns:
        test_location = df_baseline["location"].iloc[0]

        # Apply sample query
        df_filtered = ag3.phenotype_data(
            sample_sets=sample_set, sample_query=f"location == '{test_location}'"
        )

        assert isinstance(df_filtered, pd.DataFrame)
        assert len(df_filtered) > 0
        assert all(df_filtered["location"] == test_location)
    else:
        pytest.skip(
            f"No data or 'location' column found for sample set {sample_set} to test query."
        )


def test_invalid_parameters():
    """Test error handling for invalid parameters."""
    ag3 = setup_ag3()

    # Test with non-existent sample set
    with pytest.raises(
        ValueError, match=r"Sample set 'NON_EXISTENT_SAMPLE_SET' not found."
    ):
        ag3.phenotype_data(sample_sets="NON_EXISTENT_SAMPLE_SET")

    # Test with a syntactically invalid sample_query (pandas will raise ParserError)
    # The current implementation of phenotype_data catches this and warns, returning empty DF.
    with pytest.warns(UserWarning, match="Error applying sample_query"):
        df = ag3.phenotype_data(
            sample_sets="1237-VO-BJ-DJOGBENOU-VMF00050",
            sample_query="invalid query string here",
        )
        assert df.empty


def test_phenotype_binary_conversion():
    """Test binary conversion of phenotype values (internal method)."""
    ag3 = setup_ag3()

    # Create test DataFrame with mixed case phenotypes
    test_df = pd.DataFrame(
        {
            "sample_id": [
                "sample1",
                "sample2",
                "sample3",
                "sample4",
                "sample5",
                "sample6",
                "sample7",  # Add one for unmapped
            ],
            "phenotype": [
                "ALIVE",
                "Dead",
                "Resistant",
                "SUSCEPTIBLE",
                "Survived",
                "Died",
                "UNKNOWN",  # Unmapped value
            ],
        }
    )

    # Call the internal method directly
    binary_series = ag3._create_phenotype_binary_series(test_df)

    # Check results
    expected = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, np.nan]
    # Use np.testing.assert_array_equal for NaN comparison
    np.testing.assert_array_equal(binary_series.values, np.array(expected))

    # Test with invalid phenotype value
    test_df_invalid = pd.DataFrame(
        {"sample_id": ["sample1"], "phenotype": ["ANOTHER_INVALID_VALUE"]}
    )

    with pytest.warns(UserWarning, match="Unmapped phenotype values found"):
        invalid_binary = ag3._create_phenotype_binary_series(test_df_invalid)
        assert np.isnan(invalid_binary.values[0])


def test_phenotype_sample_sets():
    """Test phenotype_sample_sets method for listing available sample sets with phenotype data."""
    ag3 = setup_ag3()

    phenotype_sets = ag3.phenotype_sample_sets()

    assert isinstance(phenotype_sets, list)
    assert len(phenotype_sets) > 0

    # Test just the first one to keep test fast and ensure it returns data
    if phenotype_sets:
        sample_set_to_check = phenotype_sets[0]
        df = ag3.phenotype_data(sample_sets=sample_set_to_check)
        assert len(df) > 0
    else:
        pytest.fail("No phenotype sample sets found to test.")


def test_phenotypes_with_snp_calls():
    """Test phenotypes_with_snps method."""
    ag3 = setup_ag3()

    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"
    insecticide_query = "Deltamethrin"
    region = "2L:2420000-2430000"  # Small region to keep test fast

    ds = ag3.phenotypes_with_snps(
        sample_sets=sample_set,
        sample_query=f"insecticide == '{insecticide_query}'",
        region=region,
    )
    assert isinstance(ds, xr.Dataset)
    assert "samples" in ds.dims
    assert "variants" in ds.dims
    assert len(ds.coords["samples"]) > 0
    assert len(ds.coords["variant_position"]) > 0
    assert "call_genotype" in ds.data_vars  # Specific to SNPs
    assert "phenotype_binary" in ds.data_vars
    assert "insecticide" in ds.data_vars
    assert all(ds["insecticide"].values == insecticide_query)


def test_phenotypes_with_haplotypes():
    """Test phenotypes_with_haplotypes method."""
    ag3 = setup_ag3()

    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"
    insecticide_query = "Deltamethrin"
    region = "2L:2420000-2430000"  # Small region to keep test fast

    ds = ag3.phenotypes_with_haplotypes(
        sample_sets=sample_set,
        sample_query=f"insecticide == '{insecticide_query}'",
        region=region,
    )
    assert isinstance(ds, xr.Dataset)
    assert "samples" in ds.dims
    assert "variants" in ds.dims
    assert len(ds.coords["samples"]) > 0
    assert len(ds.coords["variant_position"]) > 0
    assert "call_genotype" in ds.data_vars  # Haplotypes also often have call_genotype
    assert "phenotype_binary" in ds.data_vars
    assert "insecticide" in ds.data_vars
    assert all(ds["insecticide"].values == insecticide_query)


def test_phenotype_data_only():
    """Test phenotype_data method returns only phenotype data (DataFrame), no genetic data."""
    ag3 = setup_ag3()

    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"
    insecticide_query = "Deltamethrin"

    # Call phenotype_data without a region or genetic data type specified
    df = ag3.phenotype_data(
        sample_sets=sample_set, sample_query=f"insecticide == '{insecticide_query}'"
    )
    assert isinstance(df, pd.DataFrame)
    assert "sample_id" in df.columns
    assert "phenotype" in df.columns
    assert "insecticide" in df.columns
    assert "dose" in df.columns
    # Assert that it's not an xarray Dataset (which would contain genetic data)
    assert not isinstance(df, xr.Dataset)
    # Also check that genetic data specific columns/attributes are NOT present
    assert "variants" not in df.columns and "variants" not in df.attrs
    assert "call_genotype" not in df.columns and "call_genotype" not in df.attrs
