import pandas as pd

from malariagen_data.anoph.frq_base import _prep_samples_for_cohort_grouping


def _make_test_df(taxon_col="taxon"):
    """Create a minimal test DataFrame for cohort grouping."""
    return pd.DataFrame(
        {
            "sample_id": ["S1", "S2", "S3"],
            taxon_col: ["gambiae", "coluzzii", "gambiae"],
            "admin1_iso": ["KE-01", "KE-01", "KE-02"],
            "year": [2020, 2020, 2021],
            "month": [1, 6, 3],
        }
    )


class TestPrepSamplesNormalizeTaxon:
    """Tests for taxon_by normalization to standard 'taxon' column. See #808."""

    def test_default_taxon_column_unchanged(self):
        """When taxon_by='taxon', no extra column is created."""
        df = _make_test_df(taxon_col="taxon")
        result = _prep_samples_for_cohort_grouping(
            df_samples=df,
            area_by="admin1_iso",
            period_by="year",
            taxon_by="taxon",
        )
        assert "taxon" in result.columns
        assert result["taxon"].iloc[0] == "gambiae"

    def test_custom_taxon_creates_standard_column(self):
        """When taxon_by is custom, a 'taxon' column is created."""
        df = _make_test_df(taxon_col="custom_taxon")
        result = _prep_samples_for_cohort_grouping(
            df_samples=df,
            area_by="admin1_iso",
            period_by="year",
            taxon_by="custom_taxon",
        )
        assert "taxon" in result.columns
        assert result["taxon"].iloc[0] == "gambiae"
        assert "custom_taxon" in result.columns

    def test_area_column_created(self):
        """area_by is normalized to 'area' column."""
        df = _make_test_df(taxon_col="taxon")
        result = _prep_samples_for_cohort_grouping(
            df_samples=df,
            area_by="admin1_iso",
            period_by="year",
            taxon_by="taxon",
        )
        assert "area" in result.columns
        assert result["area"].iloc[0] == "KE-01"
