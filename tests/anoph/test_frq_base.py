"""Tests for _prep_samples_for_cohort_grouping filter_unassigned behavior.

See: https://github.com/malariagen/malariagen-data-python/issues/806
"""

import pandas as pd

from malariagen_data.anoph.frq_base import _prep_samples_for_cohort_grouping


def _make_test_df(taxon_col="taxon"):
    """Create a test DataFrame with intermediate and unassigned taxon values."""
    return pd.DataFrame(
        {
            taxon_col: [
                "gambiae",
                "intermediate_gambcolu_arabiensis",
                "unassigned",
                "coluzzii",
            ],
            "admin1_iso": ["KE-01", "KE-01", "KE-02", "KE-02"],
            "year": [2020, 2020, 2020, 2020],
            "month": [1, 1, 1, 1],
        }
    )


class TestPrepSamplesFilterUnassigned:
    """Tests for the filter_unassigned parameter in _prep_samples_for_cohort_grouping."""

    def test_default_taxon_column_filters(self):
        """When taxon_by='taxon' and filter_unassigned=None (default),
        intermediate/unassigned values should be set to None (backward compat)."""
        df = _make_test_df(taxon_col="taxon")
        result = _prep_samples_for_cohort_grouping(
            df_samples=df,
            area_by="admin1_iso",
            period_by="year",
            taxon_by="taxon",
        )
        assert result["taxon"].iloc[0] == "gambiae"
        assert result["taxon"].iloc[1] is None
        assert result["taxon"].iloc[2] is None
        assert result["taxon"].iloc[3] == "coluzzii"

    def test_custom_column_preserves(self):
        """When taxon_by is a custom column and filter_unassigned=None (default),
        intermediate/unassigned values should be preserved."""
        df = _make_test_df(taxon_col="custom_taxon")
        result = _prep_samples_for_cohort_grouping(
            df_samples=df,
            area_by="admin1_iso",
            period_by="year",
            taxon_by="custom_taxon",
        )
        assert result["custom_taxon"].iloc[0] == "gambiae"
        assert result["custom_taxon"].iloc[1] == "intermediate_gambcolu_arabiensis"
        assert result["custom_taxon"].iloc[2] == "unassigned"
        assert result["custom_taxon"].iloc[3] == "coluzzii"

    def test_explicit_filter_true(self):
        """When filter_unassigned=True, always filter regardless of column name."""
        df = _make_test_df(taxon_col="custom_taxon")
        result = _prep_samples_for_cohort_grouping(
            df_samples=df,
            area_by="admin1_iso",
            period_by="year",
            taxon_by="custom_taxon",
            filter_unassigned=True,
        )
        assert result["custom_taxon"].iloc[0] == "gambiae"
        assert result["custom_taxon"].iloc[1] is None
        assert result["custom_taxon"].iloc[2] is None
        assert result["custom_taxon"].iloc[3] == "coluzzii"

    def test_explicit_filter_false(self):
        """When filter_unassigned=False, never filter even for default 'taxon' column."""
        df = _make_test_df(taxon_col="taxon")
        result = _prep_samples_for_cohort_grouping(
            df_samples=df,
            area_by="admin1_iso",
            period_by="year",
            taxon_by="taxon",
            filter_unassigned=False,
        )
        assert result["taxon"].iloc[0] == "gambiae"
        assert result["taxon"].iloc[1] == "intermediate_gambcolu_arabiensis"
        assert result["taxon"].iloc[2] == "unassigned"
        assert result["taxon"].iloc[3] == "coluzzii"

    def test_does_not_modify_original(self):
        """Ensure the original DataFrame is not modified."""
        df = _make_test_df(taxon_col="taxon")
        original_values = df["taxon"].tolist()
        _prep_samples_for_cohort_grouping(
            df_samples=df,
            area_by="admin1_iso",
            period_by="year",
            taxon_by="taxon",
        )
        assert df["taxon"].tolist() == original_values
