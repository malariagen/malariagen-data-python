from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
import allel  # type: ignore
from numpydoc_decorator import doc  # type: ignore
import bokeh.models
import bokeh.plotting
import bokeh.layouts
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .snp_data import AnophelesSnpData
from . import base_params, fst_params, gplt_params, plotly_params
from ..util import CacheMiss, _check_types


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
        *,
        contig,
        window_size,
        sample_sets,
        cohort1_query,
        cohort2_query,
        sample_query_options,
        site_mask,
        cohort_size,
        min_cohort_size,
        max_cohort_size,
        random_seed,
        inline_array,
        chunks,
        clip_min,
    ):
        # Compute allele counts.
        ac1 = self.snp_allele_counts(
            region=contig,
            sample_query=cohort1_query,
            sample_query_options=sample_query_options,
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
            sample_query_options=sample_query_options,
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
            with np.errstate(divide="ignore", invalid="ignore"):
                fst = allel.moving_hudson_fst(ac1, ac2, size=window_size)
                # Sometimes Fst can be very slightly below zero, clip for simplicity.
                fst = np.clip(fst, a_min=clip_min, a_max=1)
                x = allel.moving_statistic(pos, statistic=np.mean, size=window_size)

        results = dict(x=x, fst=fst)

        return results

    @_check_types
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
        sample_query_options: Optional[base_params.sample_query_options] = None,
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
        chunks: base_params.chunks = base_params.native_chunks,
        clip_min: fst_params.clip_min = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "fst_gwss_v2"

        params = dict(
            contig=contig,
            window_size=window_size,
            cohort1_query=self._prep_sample_query_param(sample_query=cohort1_query),
            cohort2_query=self._prep_sample_query_param(sample_query=cohort2_query),
            sample_query_options=sample_query_options,
            sample_sets=self._prep_sample_sets_param(sample_sets=sample_sets),
            site_mask=self._prep_optional_site_mask_param(site_mask=site_mask),
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            clip_min=clip_min,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._fst_gwss(**params, inline_array=inline_array, chunks=chunks)
            self.results_cache_set(name=name, params=params, results=results)

        x = results["x"]
        fst = results["fst"]

        return x, fst

    @_check_types
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
        sample_query_options: Optional[base_params.sample_query_options] = None,
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
        clip_min: fst_params.clip_min = 0.0,
    ) -> gplt_params.optional_figure:
        # compute Fst
        x, fst = self.fst_gwss(
            contig=contig,
            window_size=window_size,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_query_options=sample_query_options,
            sample_sets=sample_sets,
            site_mask=site_mask,
            random_seed=random_seed,
            clip_min=clip_min,
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
            y_range=(clip_min, 1),
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
        fig.yaxis.ticker = sorted(set([clip_min, 0, 1]))
        self._bokeh_style_genome_xaxis(fig, contig)

        if show:  # pragma: no cover
            bokeh.plotting.show(fig)
            return None
        else:
            return fig

    @_check_types
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
        sample_query_options: Optional[base_params.sample_query_options] = None,
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
        clip_min: fst_params.clip_min = 0.0,
        gene_labels: Optional[gplt_params.gene_labels] = None,
        gene_labelset: Optional[gplt_params.gene_labelset] = None,
    ) -> gplt_params.optional_figure:
        # gwss track
        fig1 = self.plot_fst_gwss_track(
            contig=contig,
            window_size=window_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_query_options=sample_query_options,
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
            clip_min=clip_min,
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
            gene_labels=gene_labels,
            gene_labelset=gene_labelset,
        )

        # combine plots into a single figure
        fig = bokeh.layouts.gridplot(
            [fig1, fig2],
            ncols=1,
            toolbar_location="above",
            merge_tools=True,
            sizing_mode=sizing_mode,
            toolbar_options=dict(active_inspect=None),
        )

        if show:  # pragma: no cover
            bokeh.plotting.show(fig)
            return None
        else:
            return fig

    @_check_types
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
        sample_query_options: Optional[base_params.sample_query] = None,
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
            sample_query_options=sample_query_options,
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
            sample_query_options=sample_query_options,
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

    @_check_types
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
        sample_query_options: Optional[base_params.sample_query_options] = None,
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
            sample_query_options=sample_query_options,
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

    @_check_types
    @doc(
        summary="""
            Plot one or more heatmaps of pairwise average Fst values.
        """,
        parameters=dict(
            show_se="If True, show a separate heatmap for standard errors.",
            show_zscore="If True, show a separate heatmap for Z scores (Fst / SE).",
            kwargs="Passed through to `go.Heatmap()`",
        ),
    )
    def plot_pairwise_average_fst(
        self,
        fst_df: fst_params.df_pairwise_fst,
        show_se: fst_params.show_se = False,
        show_zscore: fst_params.show_zscore = False,
        zmin: Optional[plotly_params.zmin] = 0.0,
        zmax: Optional[plotly_params.zmax] = None,
        text_auto: plotly_params.text_auto = ".3f",
        color_continuous_scale: plotly_params.color_continuous_scale = "Viridis",
        width: plotly_params.fig_width = None,
        height: plotly_params.fig_height = None,
        row_height: plotly_params.height = 50,
        col_width: plotly_params.width = 150,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        **kwargs,
    ):
        # Obtain a list of all cohorts analysed. N.B., preserve the order in
        # which the cohorts are provided in the input dataframe.
        cohorts = list(pd.unique(fst_df[["cohort1", "cohort2"]].values.flatten()))
        n = len(cohorts)

        # Helper: build a symmetric square matrix for the given statistic.
        # Diagonal is NaN (self-comparisons not applicable).
        def _build_matrix(stat: str) -> np.ndarray:
            mat = np.full((n, n), np.nan)
            cohort_idx = {c: i for i, c in enumerate(cohorts)}
            for cohort1, cohort2, fst, se in fst_df.itertuples(index=False):
                i, j = cohort_idx[cohort1], cohort_idx[cohort2]
                if stat == "fst":
                    val = fst
                elif stat == "se":
                    val = se
                else:  # Z score
                    val = np.nan if se == 0 else fst / se
                mat[i, j] = val
                mat[j, i] = val
            return mat

        # Helper: format a matrix of floats into text labels.
        def _format_text(mat: np.ndarray, fmt: str) -> List[List[str]]:
            result = []
            for row in mat:
                result.append([f"{v:{fmt}}" if np.isfinite(v) else "" for v in row])
            return result

        # Determine which panels to produce.
        plot_specs: List[tuple] = [
            ("fst", "fst", zmin, zmax, color_continuous_scale),
        ]
        if show_se:
            plot_specs.append(("se", "se", 0.0, None, color_continuous_scale))
        if show_zscore:
            plot_specs.append(("zscore", "Z score", None, None, color_continuous_scale))

        n_plots = len(plot_specs)

        # Dynamically size the figure based on number of cohorts and panels.
        panel_px = 100 + n * row_height
        if height is None:
            height = n_plots * panel_px
        if width is None:
            width = 330 + n * col_width

        # Vertical spacing as a fraction of total height; only the bottom panel
        # shows tick labels so inner gaps only need room for the next title.
        v_space = (50 / height) if n_plots > 1 else 0.0

        fig = make_subplots(
            rows=n_plots,
            cols=1,
            subplot_titles=[spec[1] for spec in plot_specs],
            vertical_spacing=v_space,
        )

        for row_idx, (stat, title, z_min, z_max, cscale) in enumerate(
            plot_specs, start=1
        ):
            mat = _build_matrix(stat)

            # Position each colorbar alongside its own subplot row.
            cb_fraction = 1.0 / n_plots
            cb_y = 1.0 - (row_idx - 0.5) * cb_fraction

            text_vals = _format_text(mat, text_auto) if text_auto else None

            heatmap = go.Heatmap(
                z=mat.tolist(),
                x=cohorts,
                y=cohorts,
                zmin=z_min,
                zmax=z_max,
                colorscale=cscale,
                text=text_vals,
                texttemplate="%{text}" if text_vals is not None else None,
                xgap=1,
                ygap=1,
                colorbar=dict(
                    y=cb_y,
                    len=cb_fraction * 0.80,
                    yanchor="middle",
                    thickness=15,
                    x=1.02,
                ),
                showscale=True,
                **kwargs,
            )
            fig.add_trace(heatmap, row=row_idx, col=1)

            # Show axis labels; x tick labels only on the bottom panel so
            # column names appear once.
            axis_suffix = "" if row_idx == 1 else str(row_idx)
            is_bottom = row_idx == n_plots
            x_title = "Cohort2" if is_bottom else ""
            fig.update_layout(
                **{
                    f"yaxis{axis_suffix}": dict(
                        title="Cohort1",
                        autorange="reversed",
                        showgrid=False,
                        linecolor="black",
                    ),
                    f"xaxis{axis_suffix}": dict(
                        title=dict(text=x_title, standoff=15),
                        showgrid=False,
                        linecolor="black",
                        tickangle=0,
                        showticklabels=is_bottom,
                    ),
                }
            )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            width=width,
            height=height,
        )
        if show:  # pragma: no cover
            fig.show(renderer=renderer)
            return None
        else:
            return fig
