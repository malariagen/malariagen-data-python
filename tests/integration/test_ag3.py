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


def test_phenotype_data():
    """Test basic functionality of phenotype_data method."""
    ag3 = setup_ag3()

    # Test with a known sample set that has phenotype data
    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"
    insecticide = "Deltamethrin"

    df = ag3.phenotype_data(sample_sets=sample_set, insecticide=insecticide)

    assert isinstance(df, pd.DataFrame)

    expected_cols = ["sample_id", "insecticide", "dose", "phenotype", "sample_set"]
    for col in expected_cols:
        assert col in df.columns

    # Check data content
    assert len(df) > 0
    assert "Deltamethrin" in df["insecticide"].unique()
    assert set(df["phenotype"].str.lower().unique()).issubset(
        {"alive", "dead", "resistant", "susceptible"}
    )
    assert sample_set in df["sample_set"].unique()


def test_phenotype_binary():
    """Test phenotype_binary method for binary outcome representation."""
    ag3 = setup_ag3()

    # Test with a known sample set that has phenotype data
    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"
    insecticide = "Deltamethrin"

    binary_series = ag3.phenotype_binary(
        sample_sets=sample_set, insecticide=insecticide
    )

    assert isinstance(binary_series, pd.Series)
    assert binary_series.name == "phenotype_binary"

    assert len(binary_series) > 0
    assert set(binary_series.unique()).issubset({0.0, 1.0, np.nan})

    # Check that values match expected binary mapping
    df = ag3.phenotype_data(sample_sets=sample_set, insecticide=insecticide)

    # Get the sample IDs from the binary series index
    sample_ids = binary_series.index.tolist()

    # Filter the DataFrame
    df_filtered = df[df["sample_id"].isin(sample_ids)]

    # Check each sample individually
    for sample_id in sample_ids:
        row = df_filtered[df_filtered["sample_id"] == sample_id]
        if len(row) > 0:
            phenotype = row["phenotype"].iloc[0].lower()
            if phenotype in ["alive", "resistant", "survived"]:
                assert binary_series.loc[sample_id] == 1.0
            elif phenotype in ["dead", "susceptible", "died"]:
                assert binary_series.loc[sample_id] == 0.0


@pytest.mark.parametrize(
    "cohort_param,expected_result",
    [
        ({"min_cohort_size": 5}, "min_size"),
        ({"max_cohort_size": 10}, "max_size"),
        ({"cohort_size": 8}, "exact_size"),
    ],
)
def test_cohort_filtering(cohort_param, expected_result):
    """Test cohort size filtering functionality."""
    ag3 = setup_ag3()

    # Test with a known sample set that has phenotype data
    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"
    insecticide = "Deltamethrin"

    df_baseline = ag3.phenotype_data(sample_sets=sample_set, insecticide=insecticide)

    # Apply cohort filtering
    df_filtered = ag3.phenotype_data(
        sample_sets=sample_set, insecticide=insecticide, **cohort_param
    )

    # Check that filtering was applied
    assert isinstance(df_filtered, pd.DataFrame)

    if expected_result == "min_size":
        cohort_keys = ["insecticide", "dose", "location", "country", "sample_set"]
        available_keys = [col for col in cohort_keys if col in df_filtered.columns]
        if available_keys:
            cohort_sizes = df_filtered.groupby(available_keys).size()
            assert all(size >= cohort_param["min_cohort_size"] for size in cohort_sizes)

    elif expected_result == "max_size":
        cohort_keys = ["insecticide", "dose", "location", "country", "sample_set"]
        available_keys = [col for col in cohort_keys if col in df_filtered.columns]
        if available_keys:
            cohort_sizes = df_filtered.groupby(available_keys).size()
            assert all(size <= cohort_param["max_cohort_size"] for size in cohort_sizes)

    elif expected_result == "exact_size":
        cohort_keys = ["insecticide", "dose", "location", "country", "sample_set"]
        available_keys = [col for col in cohort_keys if col in df_filtered.columns]
        if available_keys and len(df_filtered) > 0:
            cohort_sizes = df_filtered.groupby(available_keys).size()
            assert all(size == cohort_param["cohort_size"] for size in cohort_sizes)


@pytest.mark.parametrize(
    "param_name,param_value,expected_type",
    [
        ("insecticide", "Deltamethrin", str),
        ("insecticide", ["Deltamethrin", "Permethrin"], list),
        ("dose", 0.5, float),
        ("dose", [0.5, 1.0], list),
        ("phenotype", "alive", str),
        ("phenotype", ["alive", "dead"], list),
    ],
)
def test_parameter_validation(param_name, param_value, expected_type):
    """Test parameter validation for different input types."""
    ag3 = setup_ag3()

    # Test with a known sample set that has phenotype data
    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"

    params = {"sample_sets": sample_set, param_name: param_value}

    df = ag3.phenotype_data(**params)

    assert isinstance(df, pd.DataFrame)

    # For non-empty results
    if len(df) > 0 and param_name in df.columns:
        if isinstance(param_value, list):
            assert df[param_name].isin(param_value).all()
        else:
            assert (df[param_name] == param_value).all()


def test_sample_query():
    """Test sample_query functionality."""
    ag3 = setup_ag3()

    # Test with a known sample set that has phenotype data
    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"

    df_baseline = ag3.phenotype_data(sample_sets=sample_set)

    if len(df_baseline) > 0 and "location" in df_baseline.columns:
        test_location = df_baseline["location"].iloc[0]

        # Apply sample query
        df_filtered = ag3.phenotype_data(
            sample_sets=sample_set, sample_query=f"location == '{test_location}'"
        )

        assert isinstance(df_filtered, pd.DataFrame)
        assert len(df_filtered) > 0
        assert all(df_filtered["location"] == test_location)


def test_invalid_parameters():
    """Test error handling for invalid parameters."""
    ag3 = setup_ag3()

    # Test with invalid insecticide type
    with pytest.raises(TypeError):
        ag3.phenotype_data(sample_sets="1237-VO-BJ-DJOGBENOU-VMF00050", insecticide=123)

    # Test with invalid dose type
    with pytest.raises(TypeError):
        ag3.phenotype_data(
            sample_sets="1237-VO-BJ-DJOGBENOU-VMF00050", dose="not_a_number"
        )

    # Test with non-existent sample set
    with pytest.raises(ValueError):
        ag3.phenotype_data(sample_sets="NON_EXISTENT_SAMPLE_SET")


def test_phenotype_binary_conversion():
    """Test binary conversion of phenotype values."""
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
            ],
            "phenotype": [
                "ALIVE",
                "Dead",
                "Resistant",
                "SUSCEPTIBLE",
                "Survived",
                "Died",
            ],
        }
    )

    # Call the internal method directly
    binary_series = ag3._create_phenotype_binary_series(test_df)

    # Check results
    expected = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    assert list(binary_series.values) == expected

    # Test with invalid phenotype value
    test_df_invalid = pd.DataFrame(
        {"sample_id": ["sample1"], "phenotype": ["INVALID_VALUE"]}
    )

    with pytest.warns(UserWarning):
        invalid_binary = ag3._create_phenotype_binary_series(test_df_invalid)
        assert np.isnan(invalid_binary.values[0])


def test_phenotype_sample_sets():
    """Test phenotype_sample_sets method for listing available sample sets with phenotype data."""
    ag3 = setup_ag3()

    phenotype_sets = ag3.phenotype_sample_sets()

    # Check return type
    assert isinstance(phenotype_sets, list)

    # Check that we have at least one sample set with phenotype data
    assert len(phenotype_sets) > 0

    # Check that all returned sample sets actually have phenotype data
    for sample_set in phenotype_sets[:1]:  # Test just the first one to keep test fast
        df = ag3.phenotype_data(sample_sets=sample_set)
        assert len(df) > 0


def test_phenotypes_with_snp_calls():
    """Test phenotypes method with genetic data from snp_calls."""
    ag3 = setup_ag3()

    # Test with a known sample set that has phenotype data
    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"
    insecticide = "Deltamethrin"

    # Small region to keep test fast
    region = "2L:2420000-2430000"

    # Test with snp_calls explicitly
    ds = ag3.phenotypes(
        sample_sets=sample_set,
        insecticide=insecticide,
        region=region,
        # No analysis parameter - will default to snp_calls
    )

    # Check return type
    assert isinstance(ds, xr.Dataset)

    # Check dimensions and coordinates
    assert "samples" in ds.dims
    assert "variants" in ds.dims
    assert len(ds.coords["samples"]) > 0
    assert len(ds.coords["variant_position"]) > 0

    # Check that genetic data variables exist
    assert "call_genotype" in ds.data_vars

    # Check that phenotype data variables exist
    assert "phenotype_binary" in ds.data_vars
    assert "insecticide" in ds.data_vars


def test_phenotypes_with_haplotypes():
    """Test phenotypes method with genetic data from haplotypes."""
    ag3 = setup_ag3()

    # Test with a known sample set that has phenotype data
    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"
    insecticide = "Deltamethrin"

    # Small region to keep test fast
    region = "2L:2420000-2430000"

    # Test with haplotypes explicitly
    ds = ag3.phenotypes(
        sample_sets=sample_set,
        insecticide=insecticide,
        region=region,
        analysis="arab",  # Specify analysis to use haplotypes
    )

    # Check return type
    assert isinstance(ds, xr.Dataset)

    # Check dimensions and coordinates
    assert "samples" in ds.dims
    assert "variants" in ds.dims
    assert len(ds.coords["samples"]) > 0
    assert len(ds.coords["variant_position"]) > 0

    # Check that phenotype data variables exist
    assert "phenotype_binary" in ds.data_vars
    assert "insecticide" in ds.data_vars


def test_phenotypes_without_genetic_data():
    """Test phenotypes method without genetic data."""
    ag3 = setup_ag3()

    # Test with a known sample set that has phenotype data
    sample_set = "1237-VO-BJ-DJOGBENOU-VMF00050"
    insecticide = "Deltamethrin"

    # Test without region parameter (no genetic data)
    ds = ag3.phenotypes(sample_sets=sample_set, insecticide=insecticide)

    # Check return type
    assert isinstance(ds, xr.Dataset)

    # Check dimensions and coordinates
    assert "samples" in ds.dims
    assert "variants" not in ds.dims
    assert len(ds.coords["samples"]) > 0

    # Check that phenotype data variables exist
    assert "phenotype_binary" in ds.data_vars
    assert "insecticide" in ds.data_vars
    assert "dose" in ds.data_vars
    assert "phenotype" in ds.data_vars
