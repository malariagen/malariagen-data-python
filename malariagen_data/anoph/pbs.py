import warnings
from typing import Tuple, Optional

import numpy as np
import allel  # type: ignore
from numpydoc_decorator import doc  # type: ignore
import bokeh.models
import bokeh.plotting
import bokeh.layouts

from .snp_data import AnophelesSnpData
from . import base_params, pbs_params, gplt_params
from ..util import CacheMiss, _check_types


class AnophelesPbsAnalysis(
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

    def _pbs_gwss(
        self,
        *,
        contig,
        window_size,
        sample_sets,
        cohort1_query,
        cohort2_query,
        cohort3_query,
        sample_query_options,
        site_mask,
        cohort_size,
        min_cohort_size,
        max_cohort_size,
        normed,
        random_seed,
        inline_array,
        chunks,
        min_snps_threshold,
        window_adjustment_factor,
    ):
        # Compute allele counts for cohort 1 (focal population).
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
        # Compute allele counts for cohort 2.
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
        # Compute allele counts for cohort 3 (outgroup).
        ac3 = self.snp_allele_counts(
            region=contig,
            sample_query=cohort3_query,
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

        n_snps = len(pos)
        if n_snps < min_snps_threshold:
            raise ValueError(
                f"Too few SNP sites ({n_snps}) available for PBS GWSS. "
                f"At least {min_snps_threshold} sites are required. "
                "Try a larger genomic region or different site selection criteria."
            )
        if window_size >= n_snps:
            adjusted_window_size = max(1, n_snps // window_adjustment_factor)
            warnings.warn(
                f"window_size ({window_size}) is >= the number of SNP sites "
                f"available ({n_snps}); automatically adjusting window_size to "
                f"{adjusted_window_size} (= {n_snps} // {window_adjustment_factor}).",
                UserWarning,
                stacklevel=2,
            )
            window_size = adjusted_window_size

        with self._spinner(desc="Compute PBS"):
            with np.errstate(divide="ignore", invalid="ignore"):
                pbs = allel.pbs(
                    ac1=ac1,
                    ac2=ac2,
                    ac3=ac3,
                    window_size=window_size,
                    normed=normed,
                )
                x = allel.moving_statistic(pos, statistic=np.mean, size=window_size)

        results = dict(x=x, pbs=pbs)

        return results

    @_check_types
    @doc(
        summary="""
            Run a PBS genome-wide scan to detect lineage-specific selection
            in a focal population relative to two other populations.
            Uses the Population Branch Statistic (Yi et al. 2010).
            If window_size is >= the number of available SNP sites, a
            UserWarning is issued and window_size is automatically adjusted.
            A ValueError is raised if the number of available SNP sites is
            below min_snps_threshold.
        """,
        parameters=dict(
            min_snps_threshold="""
                Minimum number of SNP sites required. If fewer sites are
                available a ValueError is raised.
            """,
            window_adjustment_factor="""
                If window_size is >= the number of available SNP sites,
                window_size is automatically set to
                number_of_snps // window_adjustment_factor.
            """,
        ),
        returns=dict(
            x="An array containing the window centre point genomic positions.",
            pbs="An array with PBS statistic values for each window.",
        ),
    )
    def pbs_gwss(
        self,
        contig: base_params.contig,
        window_size: pbs_params.window_size,
        cohort1_query: base_params.sample_query,
        cohort2_query: base_params.sample_query,
        cohort3_query: base_params.sample_query,
        normed: pbs_params.normed = pbs_params.normed_default,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        cohort_size: Optional[base_params.cohort_size] = pbs_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = pbs_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = pbs_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
        min_snps_threshold: pbs_params.min_snps_threshold = 1000,
        window_adjustment_factor: pbs_params.window_adjustment_factor = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "pbs_gwss_v1"

        params = dict(
            contig=contig,
            window_size=window_size,
            cohort1_query=self._prep_sample_query_param(sample_query=cohort1_query),
            cohort2_query=self._prep_sample_query_param(sample_query=cohort2_query),
            cohort3_query=self._prep_sample_query_param(sample_query=cohort3_query),
            normed=normed,
            sample_query_options=sample_query_options,
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
            results = self._pbs_gwss(
                **params,
                inline_array=inline_array,
                chunks=chunks,
                min_snps_threshold=min_snps_threshold,
                window_adjustment_factor=window_adjustment_factor,
            )
            self.results_cache_set(name=name, params=params, results=results)

        x = results["x"]
        pbs = results["pbs"]

        return x, pbs

    @_check_types
    @doc(
        summary="""
            Run and plot a PBS genome-wide scan to detect lineage-specific
            selection in a focal population.
        """,
    )
    def plot_pbs_gwss_track(
        self,
        contig: base_params.contig,
        window_size: pbs_params.window_size,
        cohort1_query: base_params.sample_query,
        cohort2_query: base_params.sample_query,
        cohort3_query: base_params.sample_query,
        normed: pbs_params.normed = pbs_params.normed_default,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        cohort_size: Optional[base_params.cohort_size] = pbs_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = pbs_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = pbs_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        title: Optional[gplt_params.title] = None,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        height: gplt_params.height = 200,
        show: gplt_params.show = True,
        x_range: Optional[gplt_params.x_range] = None,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
    ) -> gplt_params.optional_figure:
        # Compute PBS.
        x, pbs = self.pbs_gwss(
            contig=contig,
            window_size=window_size,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            cohort3_query=cohort3_query,
            normed=normed,
            sample_query_options=sample_query_options,
            sample_sets=sample_sets,
            site_mask=site_mask,
            random_seed=random_seed,
        )

        # Determine X axis range.
        x_min = x[0]
        x_max = x[-1]
        if x_range is None:
            x_range = bokeh.models.Range1d(x_min, x_max, bounds="auto")

        # Create a figure.
        xwheel_zoom = bokeh.models.WheelZoomTool(
            dimensions="width", maintain_focus=False
        )
        if title is None:
            title = (
                f"Focal: {cohort1_query}\n"
                f"Comparison: {cohort2_query}\n"
                f"Outgroup: {cohort3_query}"
            )
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
            output_backend=output_backend,
        )

        # Plot PBS.
        fig.scatter(
            x=x,
            y=pbs,
            size=3,
            marker="circle",
            line_width=1,
            line_color="black",
            fill_color=None,
        )

        # Tidy up the plot.
        fig.yaxis.axis_label = "PBS"
        self._bokeh_style_genome_xaxis(fig, contig)

        if show:  # pragma: no cover
            bokeh.plotting.show(fig)
        return fig

    @_check_types
    @doc(
        summary="""
            Run and plot a PBS genome-wide scan with gene track to detect
            lineage-specific selection in a focal population.
        """,
    )
    def plot_pbs_gwss(
        self,
        contig: base_params.contig,
        window_size: pbs_params.window_size,
        cohort1_query: base_params.sample_query,
        cohort2_query: base_params.sample_query,
        cohort3_query: base_params.sample_query,
        normed: pbs_params.normed = pbs_params.normed_default,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        cohort_size: Optional[base_params.cohort_size] = pbs_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = pbs_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = pbs_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        title: Optional[gplt_params.title] = None,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        track_height: gplt_params.track_height = 190,
        genes_height: gplt_params.genes_height = gplt_params.genes_height_default,
        show: gplt_params.show = True,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
        gene_labels: Optional[gplt_params.gene_labels] = None,
        gene_labelset: Optional[gplt_params.gene_labelset] = None,
    ) -> gplt_params.optional_figure:
        # GWSS track.
        fig1 = self.plot_pbs_gwss_track(
            contig=contig,
            window_size=window_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            cohort3_query=cohort3_query,
            normed=normed,
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
        )

        fig1.xaxis.visible = False

        # Plot genes.
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

        # Combine plots into a single figure.
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
        return fig
