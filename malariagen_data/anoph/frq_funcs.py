import numpy as np
import pandas as pd


def prep_samples_for_cohort_grouping(*, df_samples, area_by, period_by):
    # Take a copy, as we will modify the dataframe.
    df_samples = df_samples.copy()

    # Fix "intermediate" or "unassigned" taxon values - we only want to build
    # cohorts with clean taxon calls, so we set other values to None.
    loc_intermediate_taxon = (
        df_samples["taxon"].str.startswith("intermediate").fillna(False)
    )
    df_samples.loc[loc_intermediate_taxon, "taxon"] = None
    loc_unassigned_taxon = (
        df_samples["taxon"].str.startswith("unassigned").fillna(False)
    )
    df_samples.loc[loc_unassigned_taxon, "taxon"] = None

    # Add period column.
    if period_by == "year":
        make_period = _make_sample_period_year
    elif period_by == "quarter":
        make_period = _make_sample_period_quarter
    elif period_by == "month":
        make_period = _make_sample_period_month
    else:  # pragma: no cover
        raise ValueError(
            f"Value for period_by parameter must be one of 'year', 'quarter', 'month'; found {period_by!r}."
        )
    sample_period = df_samples.apply(make_period, axis="columns")
    df_samples["period"] = sample_period

    # Add area column for consistent output.
    df_samples["area"] = df_samples[area_by]

    return df_samples


def build_cohorts_from_sample_grouping(*, group_samples_by_cohort, min_cohort_size):
    # Build cohorts dataframe.
    df_cohorts = group_samples_by_cohort.agg(
        size=("sample_id", len),
        lat_mean=("latitude", "mean"),
        lat_max=("latitude", "max"),
        lat_min=("latitude", "min"),
        lon_mean=("longitude", "mean"),
        lon_max=("longitude", "max"),
        lon_min=("longitude", "min"),
    )
    # Reset index so that the index fields are included as columns.
    df_cohorts = df_cohorts.reset_index()

    # Add cohort helper variables.
    cohort_period_start = df_cohorts["period"].apply(lambda v: v.start_time)
    cohort_period_end = df_cohorts["period"].apply(lambda v: v.end_time)
    df_cohorts["period_start"] = cohort_period_start
    df_cohorts["period_end"] = cohort_period_end
    # Create a label that is similar to the cohort metadata,
    # although this won't be perfect.
    df_cohorts["label"] = df_cohorts.apply(
        lambda v: f"{v.area}_{v.taxon[:4]}_{v.period}", axis="columns"
    )

    # Apply minimum cohort size.
    df_cohorts = df_cohorts.query(f"size >= {min_cohort_size}").reset_index(drop=True)

    # Early check for no cohorts.
    if len(df_cohorts) == 0:
        raise ValueError(
            "No cohorts available for the given sample selection parameters and minimum cohort size."
        )

    return df_cohorts


def add_frequency_ci(*, ds, ci_method):
    from statsmodels.stats.proportion import proportion_confint  # type: ignore

    if ci_method is not None:
        count = ds["event_count"].values
        nobs = ds["event_nobs"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            frq_ci_low, frq_ci_upp = proportion_confint(
                count=count, nobs=nobs, method=ci_method
            )
        ds["event_frequency_ci_low"] = ("variants", "cohorts"), frq_ci_low
        ds["event_frequency_ci_upp"] = ("variants", "cohorts"), frq_ci_upp


def _make_sample_period_month(row):
    year = row.year
    month = row.month
    if year > 0 and month > 0:
        return pd.Period(freq="M", year=year, month=month)
    else:
        return pd.NaT


def _make_sample_period_quarter(row):
    year = row.year
    month = row.month
    if year > 0 and month > 0:
        return pd.Period(freq="Q", year=year, month=month)
    else:
        return pd.NaT


def _make_sample_period_year(row):
    year = row.year
    if year > 0:
        return pd.Period(freq="Y", year=year)
    else:
        return pd.NaT
