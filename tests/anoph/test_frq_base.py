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


class TestPrepSamplesAreaByValidation:
    """Tests for area_by validation in _prep_samples_for_cohort_grouping."""

    def test_invalid_area_by_raises_value_error(self):
        """A non-existent area_by column should raise ValueError, not KeyError."""
        import pytest

        df = _make_test_df()
        with pytest.raises(ValueError, match="Invalid value for `area_by`"):
            _prep_samples_for_cohort_grouping(
                df_samples=df,
                area_by="nonexistent_column",
                period_by="year",
                taxon_by="taxon",
            )


class TestPrepSamplesTaxonByValidation:
    """Tests for taxon_by validation in _prep_samples_for_cohort_grouping."""

    def test_invalid_taxon_by_raises_value_error(self):
        """A non-existent taxon_by column should raise ValueError, not KeyError."""
        import pytest

        df = _make_test_df()
        with pytest.raises(ValueError, match="Invalid value for `taxon_by`"):
            _prep_samples_for_cohort_grouping(
                df_samples=df,
                area_by="admin1_iso",
                period_by="year",
                taxon_by="nonexistent_column",
            )


class TestPlotFrequenciesTimeSeriesMissingCI:
    """Tests for plot_frequencies_time_series when CI variables are absent.

    See: https://github.com/malariagen/malariagen-data-python/issues/1035
    """

    @staticmethod
    def _make_ds_without_ci():
        """Create a minimal dataset without CI variables."""
        import numpy as np
        import xarray as xr

        ds = xr.Dataset(
            {
                "variant_label": ("variants", ["V0", "V1", "V2"]),
                "cohort_taxon": ("cohorts", ["gambiae", "coluzzii"]),
                "cohort_area": ("cohorts", ["KE-01", "KE-02"]),
                "cohort_period": (
                    "cohorts",
                    pd.PeriodIndex(["2020", "2021"], freq="Y"),
                ),
                "cohort_period_start": (
                    "cohorts",
                    pd.to_datetime(["2020-01-01", "2021-01-01"]),
                ),
                "cohort_size": ("cohorts", [50, 60]),
                "event_count": (
                    ("variants", "cohorts"),
                    np.array([[10, 20], [5, 15], [25, 30]]),
                ),
                "event_nobs": (
                    ("variants", "cohorts"),
                    np.array([[100, 120], [100, 120], [100, 120]]),
                ),
                "event_frequency": (
                    ("variants", "cohorts"),
                    np.array([[0.1, 0.167], [0.05, 0.125], [0.25, 0.25]]),
                ),
            }
        )
        return ds

    @staticmethod
    def _make_ds_with_ci():
        """Create a minimal dataset with CI variables."""
        import numpy as np

        ds = TestPlotFrequenciesTimeSeriesMissingCI._make_ds_without_ci()
        ds["event_frequency_ci_low"] = (
            ("variants", "cohorts"),
            np.maximum(ds["event_frequency"].values - 0.05, 0),
        )
        ds["event_frequency_ci_upp"] = (
            ("variants", "cohorts"),
            np.minimum(ds["event_frequency"].values + 0.05, 1),
        )
        return ds

    def test_no_ci_no_error(self):
        """plot_frequencies_time_series should not raise when CI variables are absent."""
        import plotly.graph_objects as go

        from malariagen_data.anoph.frq_base import AnophelesFrequencyAnalysis

        ds = self._make_ds_without_ci()
        fig = AnophelesFrequencyAnalysis.plot_frequencies_time_series(
            None, ds, show=False
        )
        assert isinstance(fig, go.Figure)

    def test_with_ci_has_error_bars(self):
        """plot_frequencies_time_series should include error bars when CI variables are present."""
        import plotly.graph_objects as go

        from malariagen_data.anoph.frq_base import AnophelesFrequencyAnalysis

        ds = self._make_ds_with_ci()
        fig = AnophelesFrequencyAnalysis.plot_frequencies_time_series(
            None, ds, show=False
        )
        assert isinstance(fig, go.Figure)
