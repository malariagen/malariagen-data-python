"""
Visual reproduction of issue #827.
Opens two plotly graphs in your browser:
  - LEFT:  Buggy version (what the user saw — flat/zero lines)
  - RIGHT: Fixed version (correct frequencies)
"""

import numpy as np
import pandas as pd
import xarray as xr
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_mock_ds():
    """
    Cohorts (in dataset order):
      index 0: area=GH-01, taxon=gambiae,  L21F=0.00, S71P=0.00
      index 1: area=CM-SW, taxon=coluzzii, L21F=0.00, S71P=0.00
      index 2: area=BF-09, taxon=gambiae,  L21F=0.72, S71P=0.38  ← target
      index 3: area=BF-09, taxon=coluzzii, L21F=0.00, S71P=0.00
    """
    periods = pd.PeriodIndex(["2019", "2020", "2021", "2022"], freq="Y")

    ds = xr.Dataset(
        {
            "event_frequency": (
                ["variants", "cohorts"],
                np.array([
                    [0.00, 0.00, 0.72, 0.00],
                    [0.00, 0.00, 0.38, 0.00],
                ]),
            ),
            "event_count": (
                ["variants", "cohorts"],
                np.array([[0, 0, 18, 0], [0, 0, 10, 0]]),
            ),
            "event_nobs": (
                ["variants", "cohorts"],
                np.array([[25, 25, 25, 25], [25, 25, 25, 25]]),
            ),
            "event_frequency_ci_low": (["variants", "cohorts"], np.zeros((2, 4))),
            "event_frequency_ci_upp": (["variants", "cohorts"], np.ones((2, 4))),
            "variant_label":     (["variants"], np.array(["L21F", "S71P"])),
            "cohort_taxon":      (["cohorts"], np.array(["gambiae", "coluzzii", "gambiae", "coluzzii"])),
            "cohort_area":       (["cohorts"], np.array(["GH-01", "CM-SW", "BF-09", "BF-09"])),
            "cohort_period":     (["cohorts"], periods),
            "cohort_period_start": (["cohorts"], periods.start_time),
            "cohort_size":       (["cohorts"], np.array([25, 25, 25, 25])),
        }
    )
    return ds


def build_df(ds, taxa=None, areas=None, use_fix=False):
    cohort_vars = [v for v in ds if str(v).startswith("cohort_")]
    df_cohorts = ds[cohort_vars].to_dataframe()
    df_cohorts.columns = [c.split("cohort_")[1] for c in df_cohorts.columns]

    if isinstance(taxa, str):
        df_cohorts = df_cohorts[df_cohorts["taxon"] == taxa]
    if isinstance(areas, str):
        df_cohorts = df_cohorts[df_cohorts["area"] == areas]

    variant_labels = ds["variant_label"].values
    dfs = []
    for cohort_index, cohort in enumerate(df_cohorts.itertuples()):
        # Buggy: uses loop counter; Fixed: uses original index
        idx = cohort.Index if use_fix else cohort_index
        ds_cohort = ds.isel(cohorts=idx)
        for i, label in enumerate(variant_labels):
            dfs.append({
                "variant": label,
                "date": cohort.period_start,
                "frequency": float(ds_cohort["event_frequency"].values[i]),
            })

    return pd.DataFrame(dfs)


if __name__ == "__main__":
    ds = make_mock_ds()

    df_buggy = build_df(ds, taxa="gambiae", areas="BF-09", use_fix=False)
    df_fixed = build_df(ds, taxa="gambiae", areas="BF-09", use_fix=True)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "❌ BUGGY — what user saw (wrong cohort read)",
            "✅ FIXED — correct frequencies",
        ],
        shared_yaxes=True,
    )

    colors = {"L21F": "#e74c3c", "S71P": "#3498db"}

    for variant in ["L21F", "S71P"]:
        # Buggy trace
        d = df_buggy[df_buggy["variant"] == variant]
        fig.add_trace(
            go.Scatter(
                x=d["date"], y=d["frequency"],
                mode="lines+markers",
                name=f"{variant}",
                legendgroup=variant,
                line=dict(color=colors[variant]),
                marker=dict(size=10),
            ),
            row=1, col=1,
        )
        # Fixed trace
        d = df_fixed[df_fixed["variant"] == variant]
        fig.add_trace(
            go.Scatter(
                x=d["date"], y=d["frequency"],
                mode="lines+markers",
                name=f"{variant}",
                legendgroup=variant,
                showlegend=False,
                line=dict(color=colors[variant]),
                marker=dict(size=10),
            ),
            row=1, col=2,
        )

    fig.update_layout(
        title="Issue #827 — plot_frequencies_time_series<br>"
              "<sub>Filtering to area='BF-09', taxa='gambiae' | Expected: L21F=72%, S71P=38%</sub>",
        yaxis=dict(range=[-0.05, 1.05], tickformat=".0%", title="Frequency"),
        yaxis2=dict(range=[-0.05, 1.05], tickformat=".0%"),
        height=500,
        width=1000,
    )

    fig.show()