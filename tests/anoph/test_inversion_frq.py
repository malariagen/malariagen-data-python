import pytest
import pandas as pd
import xarray as xr

from malariagen_data import ag3 as _ag3
from malariagen_data.anoph.inversion_frq import AnophelesInversionFrequencyAnalysis


@pytest.fixture
def ag3_sim_api(ag3_sim_fixture):
    # Create the API with the simulator URL
    api = AnophelesInversionFrequencyAnalysis(
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

    api._test_karyotype_tag_path = ag3_sim_fixture.karyotype_tag_path
    return api


@pytest.fixture(autouse=True)
def mock_load_inversion_tags():
    from unittest.mock import patch

    def mock_load(self, inversion):
        df_tag_snps = pd.read_csv(self._test_karyotype_tag_path, sep=",")
        return df_tag_snps.query(f"inversion == '{inversion}'").reset_index()

    with patch(
        "malariagen_data.anoph.karyotype.AnophelesKaryotypeAnalysis.load_inversion_tags",
        new=mock_load,
    ):
        yield


def test_inversion_frequencies(ag3_sim_api):
    # Test valid single inversion (string)
    df = ag3_sim_api.inversion_frequencies(
        inversions="2La",
        cohorts="admin1_year",
        sample_sets="3.0",
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 3  # 3 rows (hom_ref, het, hom_alt) per inversion * per cohort
    assert list(df.columns[:3]) == ["inversion", "allele", "label"]
    assert "2La" in df["inversion"].values

    # Test valid list of inversions
    df2 = ag3_sim_api.inversion_frequencies(
        inversions=["2La", "2Rb"],
        cohorts="admin1_year",
        sample_sets="3.0",
    )
    assert isinstance(df2, pd.DataFrame)
    assert "2La" in df2["inversion"].values
    assert "2Rb" in df2["inversion"].values
    assert len(df2) == 2 * len(df)  # Two inversions compared to one

    # Test with include_counts = True
    df_counts = ag3_sim_api.inversion_frequencies(
        inversions="2La",
        cohorts="admin1_year",
        sample_sets="3.0",
        include_counts=True,
    )
    assert any([col.startswith("count_") for col in df_counts.columns])
    assert any([col.startswith("nobs_") for col in df_counts.columns])

    # Test empty list (should raise ValueError)
    with pytest.raises(ValueError):
        ag3_sim_api.inversion_frequencies(
            inversions=[],
            cohorts="admin1_year",
            sample_sets="3.0",
        )


def test_inversion_frequencies_advanced(ag3_sim_api):
    # Test valid single inversion (string)
    ds = ag3_sim_api.inversion_frequencies_advanced(
        inversions="2La",
        area_by="admin1_iso",
        period_by="year",
        sample_sets="3.0",
    )
    assert isinstance(ds, xr.Dataset)
    assert "event_frequency" in ds
    assert "event_count" in ds
    assert "event_nobs" in ds
    assert len(ds.variants) == 3  # hom_ref, het, hom_alt
    assert "2La_hom_ref" in ds.variant_label.values

    # Test valid list of inversions
    ds2 = ag3_sim_api.inversion_frequencies_advanced(
        inversions=["2La", "2Rb"],
        area_by="admin1_iso",
        period_by="year",
        sample_sets="3.0",
    )
    assert isinstance(ds2, xr.Dataset)
    assert len(ds2.variants) == 6  # 3 for 2La, 3 for 2Rb
    assert "2La_hom_ref" in ds2.variant_label.values
    assert "2Rb_hom_ref" in ds2.variant_label.values

    # Test confidence intervals
    ds_ci = ag3_sim_api.inversion_frequencies_advanced(
        inversions="2La",
        area_by="admin1_iso",
        period_by="year",
        sample_sets="3.0",
        ci_method="wilson",
    )
    assert "event_frequency_ci_low" in ds_ci
    assert "event_frequency_ci_upp" in ds_ci

    # Test empty list (should raise ValueError)
    with pytest.raises(ValueError):
        ag3_sim_api.inversion_frequencies_advanced(
            inversions=[],
            area_by="admin1_iso",
            period_by="year",
            sample_sets="3.0",
        )


@pytest.mark.parametrize("min_cohort_size", [0, 10])
def test_inversion_frequencies_min_cohort_size(ag3_sim_api, min_cohort_size):
    df = ag3_sim_api.inversion_frequencies(
        inversions="2La",
        cohorts="admin1_year",
        sample_sets="3.0",
        min_cohort_size=min_cohort_size,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 3  # At least one cohort with 3 rows
    assert list(df.columns[:3]) == ["inversion", "allele", "label"]
    # All frq_ columns should have valid frequency values
    frq_cols = [c for c in df.columns if c.startswith("frq_")]
    assert len(frq_cols) > 0
    for col in frq_cols:
        assert df[col].between(0, 1).all()


def test_inversion_frequencies_sample_query(ag3_sim_api):
    # Get the full metadata to find a valid query value.
    df_samples = ag3_sim_api.sample_metadata(sample_sets="3.0")
    # Use the first sample_set value as a query filter.
    sample_set_val = df_samples["sample_set"].iloc[0]

    df = ag3_sim_api.inversion_frequencies(
        inversions="2La",
        cohorts="admin1_year",
        sample_query=f"sample_set == '{sample_set_val}'",
        sample_sets="3.0",
        min_cohort_size=0,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 3
    assert "2La" in df["inversion"].values


def test_inversion_frequencies_sample_query_options(ag3_sim_api):
    # Get the full metadata to find a valid query value.
    df_samples = ag3_sim_api.sample_metadata(sample_sets="3.0")
    sample_set_val = df_samples["sample_set"].iloc[0]

    df = ag3_sim_api.inversion_frequencies(
        inversions="2La",
        cohorts="admin1_year",
        sample_query="sample_set == @ss_val",
        sample_query_options={
            "local_dict": {
                "ss_val": sample_set_val,
            }
        },
        sample_sets="3.0",
        min_cohort_size=0,
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) >= 3
    assert "2La" in df["inversion"].values


def test_inversion_frequencies_include_counts_columns(ag3_sim_api):
    df = ag3_sim_api.inversion_frequencies(
        inversions="2La",
        cohorts="admin1_year",
        sample_sets="3.0",
        include_counts=True,
    )
    assert isinstance(df, pd.DataFrame)

    # Check that frq_, count_, and nobs_ columns all exist
    frq_cols = [c for c in df.columns if c.startswith("frq_")]
    count_cols = [c for c in df.columns if c.startswith("count_")]
    nobs_cols = [c for c in df.columns if c.startswith("nobs_")]
    assert len(frq_cols) > 0
    assert len(count_cols) > 0
    assert len(nobs_cols) > 0

    # Cohort labels should match across frq, count, and nobs columns
    frq_labels = sorted([c.replace("frq_", "") for c in frq_cols])
    count_labels = sorted([c.replace("count_", "") for c in count_cols])
    nobs_labels = sorted([c.replace("nobs_", "") for c in nobs_cols])
    assert frq_labels == count_labels == nobs_labels

    # Counts should be non-negative integers
    for col in count_cols:
        assert (df[col] >= 0).all()


@pytest.mark.parametrize("area_by", ["country", "admin1_iso"])
def test_inversion_frequencies_advanced_area_by(ag3_sim_api, area_by):
    ds = ag3_sim_api.inversion_frequencies_advanced(
        inversions="2La",
        area_by=area_by,
        period_by="year",
        sample_sets="3.0",
        min_cohort_size=0,
    )
    assert isinstance(ds, xr.Dataset)
    assert "event_frequency" in ds
    assert "cohort_area" in ds
    assert len(ds.variants) == 3


@pytest.mark.parametrize("period_by", ["year", "month"])
def test_inversion_frequencies_advanced_period_by(ag3_sim_api, period_by):
    ds = ag3_sim_api.inversion_frequencies_advanced(
        inversions="2La",
        area_by="admin1_iso",
        period_by=period_by,
        sample_sets="3.0",
        min_cohort_size=0,
    )
    assert isinstance(ds, xr.Dataset)
    assert "event_frequency" in ds
    assert "cohort_period" in ds
    assert len(ds.variants) == 3

    # Check period values have the expected frequency string
    if period_by == "year":
        expected_freqstr = "Y-DEC"
    elif period_by == "month":
        expected_freqstr = "M"
    for p in ds["cohort_period"].values:
        assert isinstance(p, pd.Period)
        assert p.freqstr == expected_freqstr


def test_inversion_frequencies_advanced_no_cohorts(ag3_sim_api):
    # A very large min_cohort_size should result in no cohorts and raise ValueError.
    with pytest.raises(ValueError):
        ag3_sim_api.inversion_frequencies_advanced(
            inversions="2La",
            area_by="admin1_iso",
            period_by="year",
            sample_sets="3.0",
            min_cohort_size=999999,
        )
