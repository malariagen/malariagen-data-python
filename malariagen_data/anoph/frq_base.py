import numpy as np
import pandas as pd
import re
import xarray as xr
import plotly.express as px
from textwrap import dedent
from typing import Optional, Union, List
from numpydoc_decorator import doc  # type: ignore
from . import (
    plotly_params,
    frq_params,
    map_params,
)
from ..util import check_types
from .base import AnophelesBase


def prep_samples_for_cohort_grouping(*, df_samples, area_by, period_by, taxon_by):
    # Take a copy, as we will modify the dataframe.
    df_samples = df_samples.copy()

    # Fix "intermediate" or "unassigned" taxon values - we only want to build
    # cohorts with clean taxon calls, so we set other values to None.
    loc_intermediate_taxon = (
        df_samples[taxon_by].str.startswith("intermediate").fillna(False)
    )
    df_samples.loc[loc_intermediate_taxon, taxon_by] = None
    loc_unassigned_taxon = (
        df_samples[taxon_by].str.startswith("unassigned").fillna(False)
    )
    df_samples.loc[loc_unassigned_taxon, taxon_by] = None

    # Add period column.

    # Map supported period_by values to functions that return either the relevant pd.Period or pd.NaT per row.
    period_by_funcs = {
        "year": _make_sample_period_year,
        "quarter": _make_sample_period_quarter,
        "month": _make_sample_period_month,
    }

    # Get the matching function for the specified period_by value, or None.
    period_by_func = period_by_funcs.get(period_by)

    # If there were no matching functions for the specified period_by value...
    if period_by_func is None:
        # Raise a ValueError if the specified period_by value is not a column in the DataFrame.
        if period_by not in df_samples.columns:
            raise ValueError(
                f"Invalid value for `period_by`: {period_by!r}. Either specify the name of an existing column "
                "or a supported period: 'year', 'quarter', or 'month'."
            )

        # Raise a ValueError if the specified period_by column does not contain instances pd.Period.
        if not all(
            df_samples[period_by].apply(
                lambda value: pd.isnull(value) or isinstance(value, pd.Period)
            )
        ):
            raise TypeError(
                f"Invalid values in {period_by!r} column. Must be either pandas.Period or null."
            )

        # Copy the specified period_by column to a new "period" column.
        df_samples["period"] = df_samples[period_by]
    else:
        # Apply the matching period_by function to create a new "period" column.
        df_samples["period"] = df_samples.apply(period_by_func, axis="columns")

    # Copy the specified area_by column to a new "area" column.
    df_samples["area"] = df_samples[area_by]

    return df_samples


def build_cohorts_from_sample_grouping(
    *, group_samples_by_cohort, min_cohort_size, taxon_by
):
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
    if taxon_by == frq_params.taxon_by_default:
        df_cohorts["label"] = df_cohorts.apply(
            lambda v: f"{v.area}_{v[taxon_by][:4]}_{v.period}", axis="columns"
        )
    else:
        # Replace non-alphanumeric characters in the taxon with underscores.
        df_cohorts["label"] = df_cohorts.apply(
            lambda v: f"{v.area}_{re.sub(r'[^A-Za-z0-9]+', '_', str(v[taxon_by]))}_{v.period}",
            axis="columns",
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


class AnophelesFrequencyAnalysis(AnophelesBase):
    def __init__(
        self,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

    @check_types
    @doc(
        summary="""
            Plot a heatmap from a pandas DataFrame of frequencies, e.g., output
            from `snp_allele_frequencies()` or `gene_cnv_frequencies()`.
        """,
        parameters=dict(
            df="""
                A DataFrame of frequencies, e.g., output from
                `snp_allele_frequencies()` or `gene_cnv_frequencies()`.
            """,
            index="""
                One or more column headers that are present in the input dataframe.
                This becomes the heatmap y-axis row labels. The column/s must
                produce a unique index.
            """,
            max_len="""
                Displaying large styled dataframes may cause ipython notebooks to
                crash. If the input dataframe is larger than this value, an error
                will be raised.
            """,
            col_width="""
                Plot width per column in pixels (px).
            """,
            row_height="""
                Plot height per row in pixels (px).
            """,
            kwargs="""
                Passed through to `px.imshow()`.
            """,
        ),
        notes="""
            It's recommended to filter the input DataFrame to just rows of interest,
            i.e., fewer rows than `max_len`.
        """,
    )
    def plot_frequencies_heatmap(
        self,
        df: pd.DataFrame,
        index: Optional[Union[str, List[str]]] = "label",
        max_len: Optional[int] = 100,
        col_width: int = 40,
        row_height: int = 20,
        x_label: plotly_params.x_label = "Cohorts",
        y_label: plotly_params.y_label = "Variants",
        colorbar: plotly_params.colorbar = True,
        width: plotly_params.fig_width = None,
        height: plotly_params.fig_height = None,
        text_auto: plotly_params.text_auto = ".0%",
        aspect: plotly_params.aspect = "auto",
        color_continuous_scale: plotly_params.color_continuous_scale = "Reds",
        title: plotly_params.title = True,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        **kwargs,
    ) -> plotly_params.figure:
        # Check len of input.
        if max_len and len(df) > max_len:
            raise ValueError(
                dedent(
                    f"""
                Input DataFrame is longer than max_len parameter value {max_len}, which means
                that the plot is likely to be very large. If you really want to go ahead,
                please rerun the function with max_len=None.
                """
                )
            )

        # Handle title.
        if title is True:
            title = df.attrs.get("title", None)

        # Indexing.
        if index is None:
            index = [str(name) for name in df.index.names]
        df = df.reset_index().copy()
        if isinstance(index, list):
            index_col = (
                df[index]
                .astype(str)
                .apply(
                    lambda row: ", ".join([o for o in row if o is not None]),
                    axis="columns",
                )
            )
        else:
            assert isinstance(index, str)
            index_col = df[index].astype(str)

        # Check that index is unique.
        if not index_col.is_unique:
            raise ValueError(f"{index} does not produce a unique index")

        # Drop and re-order columns.
        frq_cols = [col for col in df.columns if col.startswith("frq_")]

        # Keep only freq cols.
        heatmap_df = df[frq_cols].copy()

        # Set index.
        heatmap_df.set_index(index_col, inplace=True)

        # Clean column names.
        heatmap_df.columns = heatmap_df.columns.str.lstrip("frq_")

        # Deal with width and height.
        if width is None:
            width = 400 + col_width * len(heatmap_df.columns)
            if colorbar:
                width += 40
        if height is None:
            height = 200 + row_height * len(heatmap_df)
            if title is not None:
                height += 40

        # Plotly heatmap styling.
        fig = px.imshow(
            img=heatmap_df,
            zmin=0,
            zmax=1,
            width=width,
            height=height,
            text_auto=text_auto,
            aspect=aspect,
            color_continuous_scale=color_continuous_scale,
            title=title,
            **kwargs,
        )

        fig.update_xaxes(side="bottom", tickangle=30)
        if x_label is not None:
            fig.update_xaxes(title=x_label)
        if y_label is not None:
            fig.update_yaxes(title=y_label)
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Frequency",
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
            )
        )
        if not colorbar:
            fig.update(layout_coloraxis_showscale=False)

        if show:  # pragma: no cover
            fig.show(renderer=renderer)
            return None
        else:
            return fig

    @check_types
    @doc(
        summary="Create a time series plot of variant frequencies using plotly.",
        parameters=dict(
            ds="""
                A dataset of variant frequencies, such as returned by
                `snp_allele_frequencies_advanced()`,
                `aa_allele_frequencies_advanced()` or
                `gene_cnv_frequencies_advanced()`.
            """,
            kwargs="Passed through to `px.line()`.",
        ),
        returns="""
            A plotly figure containing line graphs. The resulting figure will
            have one panel per cohort, grouped into columns by taxon, and
            grouped into rows by area. Markers and lines show frequencies of
            variants.
        """,
    )
    def plot_frequencies_time_series(
        self,
        ds: xr.Dataset,
        height: plotly_params.fig_height = None,
        width: plotly_params.fig_width = None,
        title: plotly_params.title = True,
        legend_sizing: plotly_params.legend_sizing = "constant",
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        taxa: frq_params.taxa = None,
        areas: frq_params.areas = None,
        **kwargs,
    ) -> plotly_params.figure:
        # Handle title.
        if title is True:
            title = ds.attrs.get("title", None)

        # Extract cohorts into a dataframe.
        cohort_vars = [v for v in ds if str(v).startswith("cohort_")]
        df_cohorts = ds[cohort_vars].to_dataframe()
        df_cohorts.columns = [c.split("cohort_")[1] for c in df_cohorts.columns]  # type: ignore

        # If specified, restrict the dataframe by taxa.
        if isinstance(taxa, str):
            df_cohorts = df_cohorts[df_cohorts["taxon"] == taxa]
        elif isinstance(taxa, (list, tuple)):
            df_cohorts = df_cohorts[df_cohorts["taxon"].isin(taxa)]

        # If specified, restrict the dataframe by areas.
        if isinstance(areas, str):
            df_cohorts = df_cohorts[df_cohorts["area"] == areas]
        elif isinstance(areas, (list, tuple)):
            df_cohorts = df_cohorts[df_cohorts["area"].isin(areas)]

        # Extract variant labels.
        variant_labels = ds["variant_label"].values

        # Build a long-form dataframe from the dataset.
        dfs = []
        for cohort_index, cohort in enumerate(df_cohorts.itertuples()):
            ds_cohort = ds.isel(cohorts=cohort_index)
            df = pd.DataFrame(
                {
                    "taxon": cohort.taxon,
                    "area": cohort.area,
                    "date": cohort.period_start,
                    "period": str(
                        cohort.period
                    ),  # use string representation for hover label
                    "sample_size": cohort.size,
                    "variant": variant_labels,
                    "count": ds_cohort["event_count"].values,
                    "nobs": ds_cohort["event_nobs"].values,
                    "frequency": ds_cohort["event_frequency"].values,
                    "frequency_ci_low": ds_cohort["event_frequency_ci_low"].values,
                    "frequency_ci_upp": ds_cohort["event_frequency_ci_upp"].values,
                }
            )
            dfs.append(df)
        df_events = pd.concat(dfs, axis=0).reset_index(drop=True)

        # Remove events with no observations.
        df_events = df_events.query("nobs > 0").copy()

        # Calculate error bars.
        frq = df_events["frequency"]
        frq_ci_low = df_events["frequency_ci_low"]
        frq_ci_upp = df_events["frequency_ci_upp"]
        df_events["frequency_error"] = frq_ci_upp - frq
        df_events["frequency_error_minus"] = frq - frq_ci_low

        # Make a plot.
        fig = px.line(
            df_events,
            facet_col="taxon",
            facet_row="area",
            x="date",
            y="frequency",
            error_y="frequency_error",
            error_y_minus="frequency_error_minus",
            color="variant",
            markers=True,
            hover_name="variant",
            hover_data={
                "frequency": ":.0%",
                "period": True,
                "area": True,
                "taxon": True,
                "sample_size": True,
                "date": False,
                "variant": False,
            },
            height=height,
            width=width,
            title=title,
            labels={
                "date": "Date",
                "frequency": "Frequency",
                "variant": "Variant",
                "taxon": "Taxon",
                "area": "Area",
                "period": "Period",
                "sample_size": "Sample size",
            },
            **kwargs,
        )

        # Tidy plot.
        fig.update_layout(
            yaxis_range=[-0.05, 1.05],
            legend=dict(itemsizing=legend_sizing, tracegroupgap=0),
        )

        if show:  # pragma: no cover
            fig.show(renderer=renderer)
            return None
        else:
            return fig

    @check_types
    @doc(
        summary="""
            Plot markers on a map showing variant frequencies for cohorts grouped
            by area (space), period (time) and taxon.
        """,
        parameters=dict(
            m="The map on which to add the markers.",
            variant="Index or label of variant to plot.",
            taxon="Taxon to show markers for.",
            period="Time period to show markers for.",
            clear="""
                If True, clear all layers (except the base layer) from the map
                before adding new markers.
            """,
        ),
    )
    def plot_frequencies_map_markers(
        self,
        m,
        ds: frq_params.ds_frequencies_advanced,
        variant: Union[int, str],
        taxon: str,
        period: pd.Period,
        clear: bool = True,
    ):
        # Only import here because of some problems importing globally.
        import ipyleaflet  # type: ignore
        import ipywidgets  # type: ignore

        # Slice dataset to variant of interest.
        if isinstance(variant, int):
            ds_variant = ds.isel(variants=variant)
            variant_label = ds["variant_label"].values[variant]
        else:
            assert isinstance(variant, str)
            ds_variant = ds.set_index(variants="variant_label").sel(variants=variant)
            variant_label = variant

        # Convert to a dataframe for convenience.
        df_markers = ds_variant[
            [
                "cohort_taxon",
                "cohort_area",
                "cohort_period",
                "cohort_lat_mean",
                "cohort_lon_mean",
                "cohort_size",
                "event_frequency",
                "event_frequency_ci_low",
                "event_frequency_ci_upp",
            ]
        ].to_dataframe()

        # Select data matching taxon and period parameters.
        df_markers = df_markers.loc[
            (
                (df_markers["cohort_taxon"] == taxon)
                & (df_markers["cohort_period"] == period)
            )
        ]

        # Clear existing layers in the map.
        if clear:
            for layer in m.layers[1:]:
                m.remove_layer(layer)

        # Add markers.
        for x in df_markers.itertuples():
            marker = ipyleaflet.CircleMarker()
            marker.location = (x.cohort_lat_mean, x.cohort_lon_mean)
            marker.radius = 20
            marker.color = "black"
            marker.weight = 1
            marker.fill_color = "red"
            marker.fill_opacity = x.event_frequency
            popup_html = f"""
                <strong>{variant_label}</strong> <br/>
                Taxon: {x.cohort_taxon} <br/>
                Area: {x.cohort_area} <br/>
                Period: {x.cohort_period} <br/>
                Sample size: {x.cohort_size} <br/>
                Frequency: {x.event_frequency:.0%}
                (95% CI: {x.event_frequency_ci_low:.0%} - {x.event_frequency_ci_upp:.0%})
            """
            marker.popup = ipyleaflet.Popup(
                child=ipywidgets.HTML(popup_html),
                auto_pan=False,
            )
            m.add(marker)

    @check_types
    @doc(
        summary="""
            Create an interactive map with markers showing variant frequencies or
            cohorts grouped by area (space), period (time) and taxon.
        """,
        parameters=dict(
            title="""
                If True, attempt to use metadata from input dataset as a plot
                title. Otherwise, use supplied value as a title.
            """,
            epilogue="Additional text to display below the map.",
        ),
        returns="""
            An interactive map with widgets for selecting which variant, taxon
            and time period to display.
        """,
    )
    def plot_frequencies_interactive_map(
        self,
        ds: frq_params.ds_frequencies_advanced,
        center: map_params.center = map_params.center_default,
        zoom: map_params.zoom = map_params.zoom_default,
        title: Optional[Union[bool, str]] = True,
        epilogue: Union[bool, str] = True,
    ):
        import ipyleaflet
        import ipywidgets

        # Handle title.
        if title is True:
            title = ds.attrs.get("title", None)

        # Create a map.
        freq_map = ipyleaflet.Map(center=center, zoom=zoom)

        # Set up interactive controls.
        variants = ds["variant_label"].values
        taxa = ds["cohort_taxon"].to_pandas().dropna().unique()  # type: ignore
        periods = ds["cohort_period"].to_pandas().dropna().unique()  # type: ignore
        controls = ipywidgets.interactive(
            self.plot_frequencies_map_markers,
            m=ipywidgets.fixed(freq_map),
            ds=ipywidgets.fixed(ds),
            variant=ipywidgets.Dropdown(options=variants, description="Variant: "),
            taxon=ipywidgets.Dropdown(options=taxa, description="Taxon: "),
            period=ipywidgets.Dropdown(options=periods, description="Period: "),
            clear=ipywidgets.fixed(True),
        )

        # Lay out widgets.
        components = []
        if title is not None:
            components.append(ipywidgets.HTML(value=f"<h3>{title}</h3>"))
        components.append(controls)
        components.append(freq_map)
        if epilogue is True:
            epilogue = """
                Variant frequencies are shown as coloured markers. Opacity of color
                denotes frequency. Click on a marker for more information.
            """
        if epilogue:
            components.append(ipywidgets.HTML(value=f"{epilogue}"))

        out = ipywidgets.VBox(components)

        return out
