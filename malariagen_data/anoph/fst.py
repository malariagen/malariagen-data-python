from typing import Tuple, Optional

import numpy as np
import pandas as pd
import allel  # type: ignore
from numpydoc_decorator import doc  # type: ignore
import bokeh.models
import bokeh.plotting
import bokeh.layouts
import plotly.express as px

from .snp_data import AnophelesSnpData
from . import base_params, fst_params, gplt_params, plotly_params
from ..util import CacheMiss, check_types


class AnophelesFstAnalysis(
    AnophelesSnpData,
):
    def __init__(
        self,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

    def _fst_gwss(
        self,
        contig,
        window_size,
        sample_sets,
        cohort1_query,
        cohort2_query,
        site_mask,
        cohort_size,
        min_cohort_size,
        max_cohort_size,
        random_seed,
        inline_array,
        chunks,
    ):
        # Compute allele counts.
        ac1 = self.snp_allele_counts(
            region=contig,
            sample_query=cohort1_query,
            sample_sets=sample_sets,
            site_mask=site_mask,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            inline_array=inline_array,
            chunks=chunks,
        )
        ac2 = self.snp_allele_counts(
            region=contig,
            sample_query=cohort2_query,
            sample_sets=sample_sets,
            site_mask=site_mask,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            inline_array=inline_array,
            chunks=chunks,
        )

        with self._spinner(desc="Load SNP positions"):
            pos = self.snp_sites(
                region=contig,
                field="POS",
                site_mask=site_mask,
                inline_array=inline_array,
                chunks=chunks,
            ).compute()

        with self._spinner(desc="Compute Fst"):
            fst = allel.moving_hudson_fst(ac1, ac2, size=window_size)
            # Sometimes Fst can be very slightly below zero, clip for simplicity.
            fst = np.clip(fst, a_min=0, a_max=1)
            x = allel.moving_statistic(pos, statistic=np.mean, size=window_size)

        results = dict(x=x, fst=fst)

        return results

    @check_types
    @doc(
        summary="""
            Run a Fst genome-wide scan to investigate genetic differentiation
            between two cohorts.
        """,
        returns=dict(
            x="An array containing the window centre point genomic positions",
            fst="An array with Fst statistic values for each window.",
        ),
    )
    def fst_gwss(
        self,
        contig: base_params.contig,
        window_size: fst_params.window_size,
        cohort1_query: base_params.sample_query,
        cohort2_query: base_params.sample_query,
        sample_sets: Optional[base_params.sample_sets] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        cohort_size: Optional[base_params.cohort_size] = fst_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = fst_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = fst_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "fst_gwss_v2"

        params = dict(
            contig=contig,
            window_size=window_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_sets=self._prep_sample_sets_param(sample_sets=sample_sets),
            site_mask=self._prep_optional_site_mask_param(site_mask=site_mask),
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._fst_gwss(**params, inline_array=inline_array, chunks=chunks)
            self.results_cache_set(name=name, params=params, results=results)

        x = results["x"]
        fst = results["fst"]

        return x, fst

    @check_types
    @doc(
        summary="""
            Run and plot a Fst genome-wide scan to investigate genetic
            differentiation between two cohorts.
        """,
    )
    def plot_fst_gwss_track(
        self,
        contig: base_params.contig,
        window_size: fst_params.window_size,
        cohort1_query: base_params.sample_query,
        cohort2_query: base_params.sample_query,
        sample_sets: Optional[base_params.sample_sets] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        cohort_size: Optional[base_params.cohort_size] = fst_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = fst_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = fst_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        title: Optional[gplt_params.title] = None,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        height: gplt_params.height = 200,
        show: gplt_params.show = True,
        x_range: Optional[gplt_params.x_range] = None,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
    ) -> gplt_params.figure:
        # compute Fst
        x, fst = self.fst_gwss(
            contig=contig,
            window_size=window_size,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_sets=sample_sets,
            site_mask=site_mask,
            random_seed=random_seed,
        )

        # determine X axis range
        x_min = x[0]
        x_max = x[-1]
        if x_range is None:
            x_range = bokeh.models.Range1d(x_min, x_max, bounds="auto")

        # create a figure
        xwheel_zoom = bokeh.models.WheelZoomTool(
            dimensions="width", maintain_focus=False
        )
        if title is None:
            title = f"Cohort 1: {cohort1_query}\nCohort 2: {cohort2_query}"
        fig = bokeh.plotting.figure(
            title=title,
            tools=[
                "xpan",
                "xzoom_in",
                "xzoom_out",
                xwheel_zoom,
                "reset",
                "save",
                "crosshair",
            ],
            active_inspect=None,
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            toolbar_location="above",
            x_range=x_range,
            y_range=(0, 1),
            output_backend=output_backend,
        )

        # plot Fst
        fig.scatter(
            x=x,
            y=fst,
            size=3,
            marker="circle",
            line_width=1,
            line_color="black",
            fill_color=None,
        )

        # tidy up the plot
        fig.yaxis.axis_label = "Fst"
        fig.yaxis.ticker = [0, 1]
        self._bokeh_style_genome_xaxis(fig, contig)

        if show:  # pragma: no cover
            bokeh.plotting.show(fig)
            return None
        else:
            return fig

    @check_types
    @doc(
        summary="""
            Run and plot a Fst genome-wide scan to investigate genetic
            differentiation between two cohorts.
        """,
    )
    def plot_fst_gwss(
        self,
        contig: base_params.contig,
        window_size: fst_params.window_size,
        cohort1_query: base_params.sample_query,
        cohort2_query: base_params.sample_query,
        sample_sets: Optional[base_params.sample_sets] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        cohort_size: Optional[base_params.cohort_size] = fst_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = fst_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = fst_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        title: Optional[gplt_params.title] = None,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        track_height: gplt_params.track_height = 190,
        genes_height: gplt_params.genes_height = gplt_params.genes_height_default,
        show: gplt_params.show = True,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
    ) -> gplt_params.figure:
        # gwss track
        fig1 = self.plot_fst_gwss_track(
            contig=contig,
            window_size=window_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_sets=sample_sets,
            site_mask=site_mask,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            title=title,
            sizing_mode=sizing_mode,
            width=width,
            height=track_height,
            show=False,
            output_backend=output_backend,
        )

        fig1.xaxis.visible = False

        # plot genes
        fig2 = self.plot_genes(
            region=contig,
            sizing_mode=sizing_mode,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
            output_backend=output_backend,
        )

        # combine plots into a single figure
        fig = bokeh.layouts.gridplot(
            [fig1, fig2],
            ncols=1,
            toolbar_location="above",
            merge_tools=True,
            sizing_mode=sizing_mode,
        )

        if show:  # pragma: no cover
            bokeh.plotting.show(fig)
            return None
        else:
            return fig

    @check_types
    @doc(
        summary="""
            Compute average Hudson's Fst between two specified cohorts.
        """,
        returns="""
            A NumPy float of the Fst value and the standard error (SE).
        """,
    )
    def average_fst(
        self,
        region: base_params.region,
        cohort1_query: base_params.sample_query,
        cohort2_query: base_params.sample_query,
        sample_sets: Optional[base_params.sample_sets] = None,
        cohort_size: Optional[base_params.cohort_size] = fst_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = fst_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = fst_params.max_cohort_size_default,
        n_jack: base_params.n_jack = 200,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        site_class: Optional[base_params.site_class] = None,
        random_seed: base_params.random_seed = 42,
    ) -> Tuple[float, float]:
        # Calculate allele counts for each cohort.
        ac1 = self.snp_allele_counts(
            region=region,
            sample_sets=sample_sets,
            sample_query=cohort1_query,
            cohort_size=cohort_size,
            site_mask=site_mask,
            site_class=site_class,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )
        ac2 = self.snp_allele_counts(
            region=region,
            sample_sets=sample_sets,
            sample_query=cohort2_query,
            cohort_size=cohort_size,
            site_mask=site_mask,
            site_class=site_class,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        # Calculate block length for jackknife.
        n_sites = ac1.shape[0]  # number of sites
        block_length = n_sites // n_jack  # number of sites in each block

        # Calculate average Fst.
        fst, se, _, _ = allel.blockwise_hudson_fst(ac1, ac2, blen=block_length)

        # Normalise to Python scalar types.
        fst = float(fst)
        se = float(se)

        # Fst estimate can sometimes be slightly negative, but clip at
        # zero.
        if fst < 0:
            fst = 0.0

        return fst, se

    @check_types
    @doc(
        summary="""
            Compute pairwise average Hudson's Fst between a set of specified cohorts.
        """,
    )
    def pairwise_average_fst(
        self,
        region: base_params.region,
        cohorts: base_params.cohorts,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        cohort_size: Optional[base_params.cohort_size] = fst_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = fst_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = fst_params.max_cohort_size_default,
        n_jack: base_params.n_jack = 200,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        site_class: Optional[base_params.site_class] = None,
        random_seed: base_params.random_seed = 42,
    ) -> fst_params.df_pairwise_fst:
        # Set up cohort queries.
        cohorts_checked = self._setup_cohort_queries(
            cohorts,
            sample_sets=sample_sets,
            sample_query=sample_query,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
        )

        cohort_ids = list(cohorts_checked.keys())
        cohort_queries = list(cohorts_checked.values())
        cohort1_ids = []
        cohort2_ids = []
        fst_stats = []
        se_stats = []

        n_cohorts = len(cohorts_checked)
        for i in range(n_cohorts):
            for j in range(i + 1, n_cohorts):
                (fst, se) = self.average_fst(
                    region=region,
                    cohort1_query=cohort_queries[i],
                    cohort2_query=cohort_queries[j],
                    sample_sets=sample_sets,
                    cohort_size=cohort_size,
                    min_cohort_size=min_cohort_size,
                    max_cohort_size=max_cohort_size,
                    n_jack=n_jack,
                    site_mask=site_mask,
                    site_class=site_class,
                    random_seed=random_seed,
                )
                cohort1_ids.append(cohort_ids[i])
                cohort2_ids.append(cohort_ids[j])
                fst_stats.append(fst)
                se_stats.append(se)

        fst_df = pd.DataFrame(
            {
                "cohort1": cohort1_ids,
                "cohort2": cohort2_ids,
                "fst": fst_stats,
                "se": se_stats,
            }
        )

        return fst_df

    @check_types
    @doc(
        summary="""
            Plot a heatmap of pairwise average Fst values.
        """,
        parameters=dict(
            annotate_se="If True, show standard error values in the upper triangle of the plot.",
            kwargs="Passed through to `px.imshow()`",
        ),
    )
    def plot_pairwise_average_fst(
        self,
        fst_df: fst_params.df_pairwise_fst,
        annotation: fst_params.annotation = None,
        zmin: Optional[plotly_params.zmin] = 0.0,
        zmax: Optional[plotly_params.zmax] = None,
        text_auto: plotly_params.text_auto = ".3f",
        color_continuous_scale: plotly_params.color_continuous_scale = "gray_r",
        width: plotly_params.fig_width = 700,
        height: plotly_params.fig_height = 600,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        **kwargs,
    ):
        # setup df
        cohort_list = np.unique(fst_df[["cohort1", "cohort2"]].values)
        # df to fill
        fig_df = pd.DataFrame(columns=cohort_list, index=cohort_list)
        # fill df from fst_df
        for index_key in range(len(fst_df)):
            index = fst_df.iloc[index_key]["cohort1"]
            col = fst_df.iloc[index_key]["cohort2"]
            fst = fst_df.iloc[index_key]["fst"]
            fig_df[index][col] = fst
            if annotation == "standard error":
                se = fst_df.iloc[index_key]["se"]
                fig_df[col][index] = se
            elif annotation == "Z score":
                zs = fst_df.iloc[index_key]["fst"] / fst_df.iloc[index_key]["se"]
                fig_df[col][index] = zs
            else:
                fig_df.loc[index, col] = fst

        # create plot
        with np.errstate(invalid="ignore"):
            fig = px.imshow(
                img=fig_df,
                zmin=zmin,
                zmax=zmax,
                width=width,
                height=height,
                text_auto=text_auto,
                color_continuous_scale=color_continuous_scale,
                aspect="auto",
                **kwargs,
            )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)")
        fig.update_yaxes(showgrid=False, linecolor="black")
        fig.update_xaxes(showgrid=False, linecolor="black")

        if show:  # pragma: no cover
            fig.show(renderer=renderer)
            return None
        else:
            return fig
