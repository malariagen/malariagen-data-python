from typing import Optional, Tuple

import allel  # type: ignore
import bokeh.layouts
import bokeh.models
import bokeh.palettes
import bokeh.plotting
import numpy as np
from numpydoc_decorator import doc  # type: ignore

from .hap_data import AnophelesHapData, hap_params
from ..util import CacheMiss, _check_types
from . import base_params, gplt_params, xpehh_params


class AnophelesXpehhAnalysis(
    AnophelesHapData,
):
    def __init__(
        self,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

    @property
    def _xpehh_gwss_cache_name(self):
        return "xpehh_gwss_v1"

    def _get_xpehh_gwss_cache_name(self):
        """Safely resolve the xpehh gwss cache name.

        Supports class attribute, property, or legacy method override.
        Falls back to the default "xpehh_gwss_v1" if resolution fails.

        See also: https://github.com/malariagen/malariagen-data-python/issues/1151
        """
        try:
            name = self._xpehh_gwss_cache_name
            # Handle legacy case where _xpehh_gwss_cache_name might be a
            # callable method rather than a property or class attribute.
            if callable(name):
                name = name()
            if isinstance(name, str) and len(name) > 0:
                return name
        except NotImplementedError:
            pass
        # Fallback to default.
        return "xpehh_gwss_v1"

    @_check_types
    @doc(
        summary="Run XP-EHH GWSS.",
        returns=dict(
            x="An array containing the window centre point genomic positions.",
            xpehh="An array with XP-EHH statistic values for each window.",
        ),
    )
    def xpehh_gwss(
        self,
        contig: base_params.contig,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        cohort1_query: Optional[base_params.sample_query] = None,
        cohort2_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        window_size: xpehh_params.window_size = xpehh_params.window_size_default,
        percentiles: xpehh_params.percentiles = xpehh_params.percentiles_default,
        filter_min_maf: xpehh_params.filter_min_maf = xpehh_params.filter_min_maf_default,
        map_pos: Optional[xpehh_params.map_pos] = None,
        min_ehh: xpehh_params.min_ehh = xpehh_params.min_ehh_default,
        max_gap: xpehh_params.max_gap = xpehh_params.max_gap_default,
        gap_scale: xpehh_params.gap_scale = xpehh_params.gap_scale_default,
        include_edges: xpehh_params.include_edges = True,
        use_threads: xpehh_params.use_threads = True,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = xpehh_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = xpehh_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data
        name = self._get_xpehh_gwss_cache_name()

        params = dict(
            contig=contig,
            analysis=self._prep_phasing_analysis_param(analysis=analysis),
            window_size=window_size,
            percentiles=percentiles,
            filter_min_maf=filter_min_maf,
            map_pos=map_pos,
            min_ehh=min_ehh,
            include_edges=include_edges,
            max_gap=max_gap,
            gap_scale=gap_scale,
            use_threads=use_threads,
            sample_sets=self._prep_sample_sets_param(sample_sets=sample_sets),
            # N.B., do not be tempted to convert this sample query into integer
            # indices using _prep_sample_selection_params, because the indices
            # are different in the haplotype data.
            cohort1_query=self._prep_sample_query_param(sample_query=cohort1_query),
            cohort2_query=self._prep_sample_query_param(sample_query=cohort2_query),
            sample_query_options=sample_query_options,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._xpehh_gwss(
                chunks=chunks, inline_array=inline_array, **params
            )
            self.results_cache_set(name=name, params=params, results=results)

        x = results["x"]
        xpehh = results["xpehh"]

        return x, xpehh

    def _xpehh_gwss(
        self,
        *,
        contig,
        analysis,
        sample_sets,
        cohort1_query,
        cohort2_query,
        sample_query_options,
        window_size,
        percentiles,
        filter_min_maf,
        map_pos,
        min_ehh,
        max_gap,
        gap_scale,
        include_edges,
        use_threads,
        min_cohort_size,
        max_cohort_size,
        random_seed,
        chunks,
        inline_array,
    ):
        ds_haps1 = self.haplotypes(
            region=contig,
            analysis=analysis,
            sample_query=cohort1_query,
            sample_query_options=sample_query_options,
            sample_sets=sample_sets,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            chunks=chunks,
            inline_array=inline_array,
        )

        ds_haps2 = self.haplotypes(
            region=contig,
            analysis=analysis,
            sample_query=cohort2_query,
            sample_query_options=sample_query_options,
            sample_sets=sample_sets,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            chunks=chunks,
            inline_array=inline_array,
        )

        gt1 = allel.GenotypeDaskArray(ds_haps1["call_genotype"].data)
        gt2 = allel.GenotypeDaskArray(ds_haps2["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes for cohort 1"):
            ht1 = gt1.to_haplotypes().compute()
        with self._dask_progress(desc="Load haplotypes for cohort 2"):
            ht2 = gt2.to_haplotypes().compute()

        with self._spinner("Compute XPEHH"):
            ac1 = ht1.count_alleles(max_allele=1)
            ac2 = ht2.count_alleles(max_allele=1)
            pos = ds_haps1["variant_position"].values

            if filter_min_maf > 0:
                ac = ac1 + ac2
                af = ac.to_frequencies()
                maf = np.min(af, axis=1)
                maf_filter = maf > filter_min_maf

                ht1 = ht1.compress(maf_filter, axis=0)
                ht2 = ht2.compress(maf_filter, axis=0)
                pos = pos[maf_filter]

            # compute XP-EHH
            xp = allel.xpehh(
                h1=ht1,
                h2=ht2,
                pos=pos,
                map_pos=map_pos,
                min_ehh=min_ehh,
                include_edges=include_edges,
                max_gap=max_gap,
                gap_scale=gap_scale,
                use_threads=use_threads,
            )

            # remove any NaNs
            na_mask = ~np.isnan(xp)
            xp = xp[na_mask]
            pos = pos[na_mask]

            if window_size:
                # guard against window_size exceeding available data
                if xp.shape[0] < window_size:
                    raise ValueError(
                        f"Not enough values ({xp.shape[0]}) for window size "
                        f"({window_size}). Please reduce the window size or "
                        f"use different filtering criteria."
                    )
                xp = allel.moving_statistic(
                    xp, statistic=np.percentile, size=window_size, q=percentiles
                )
                pos = allel.moving_statistic(pos, statistic=np.mean, size=window_size)

        results = dict(x=pos, xpehh=xp)

        return results

    @doc(
        summary="Run and plot XP-EHH GWSS data.",
    )
    def plot_xpehh_gwss_track(
        self,
        contig: base_params.contig,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        cohort1_query: Optional[base_params.sample_query] = None,
        cohort2_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        window_size: xpehh_params.window_size = xpehh_params.window_size_default,
        percentiles: xpehh_params.percentiles = xpehh_params.percentiles_default,
        filter_min_maf: xpehh_params.filter_min_maf = xpehh_params.filter_min_maf_default,
        map_pos: Optional[xpehh_params.map_pos] = None,
        min_ehh: xpehh_params.min_ehh = xpehh_params.min_ehh_default,
        max_gap: xpehh_params.max_gap = xpehh_params.max_gap_default,
        gap_scale: xpehh_params.gap_scale = xpehh_params.gap_scale_default,
        include_edges: xpehh_params.include_edges = True,
        use_threads: xpehh_params.use_threads = True,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = xpehh_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = xpehh_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        palette: xpehh_params.palette = xpehh_params.palette_default,
        title: Optional[gplt_params.title] = None,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        height: gplt_params.height = 200,
        show: gplt_params.show = True,
        x_range: Optional[gplt_params.x_range] = None,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> gplt_params.optional_figure:
        # compute xpehh
        x, xpehh = self.xpehh_gwss(
            contig=contig,
            analysis=analysis,
            window_size=window_size,
            percentiles=percentiles,
            filter_min_maf=filter_min_maf,
            map_pos=map_pos,
            min_ehh=min_ehh,
            max_gap=max_gap,
            gap_scale=gap_scale,
            include_edges=include_edges,
            use_threads=use_threads,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_query_options=sample_query_options,
            sample_sets=sample_sets,
            random_seed=random_seed,
            chunks=chunks,
            inline_array=inline_array,
        )

        if len(x) == 0:
            raise ValueError(
                "No XP-EHH values remain after filtering. "
                "Try relaxing filter_min_maf or min_ehh parameters."
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
            if cohort1_query is None or cohort2_query is None:
                title = "XP-EHH"
            else:
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
            output_backend=output_backend,
        )

        if window_size:
            if isinstance(percentiles, int):
                percentiles = (percentiles,)
            # Ensure percentiles are sorted so that colors make sense.
            percentiles = tuple(sorted(percentiles))

        # add an empty dimension to XP-EHH array if 1D
        xpehh = np.reshape(xpehh, (xpehh.shape[0], -1))

        # select the base color palette to work from
        base_palette = bokeh.palettes.all_palettes[palette][8]

        # keep only enough colours to plot the XP-EHH tracks
        bokeh_palette = base_palette[: xpehh.shape[1]]

        # reverse the colors so darkest is last
        bokeh_palette = bokeh_palette[::-1]

        for i in range(xpehh.shape[1]):
            xpehh_perc = xpehh[:, i]
            color = bokeh_palette[i]

            # plot XP-EHH
            fig.circle(
                x=x,
                y=xpehh_perc,
                size=4,
                line_width=0,
                line_color=color,
                fill_color=color,
            )

        # tidy up the plot
        fig.yaxis.axis_label = "XP-EHH"
        self._bokeh_style_genome_xaxis(fig, contig)

        if show:  # pragma: no cover
            bokeh.plotting.show(fig)
            return None
        else:
            return fig

    @_check_types
    @doc(
        summary="Run and plot XP-EHH GWSS data.",
    )
    def plot_xpehh_gwss(
        self,
        contig: base_params.contig,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        cohort1_query: Optional[base_params.sample_query] = None,
        cohort2_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        window_size: xpehh_params.window_size = xpehh_params.window_size_default,
        percentiles: xpehh_params.percentiles = xpehh_params.percentiles_default,
        filter_min_maf: xpehh_params.filter_min_maf = xpehh_params.filter_min_maf_default,
        map_pos: Optional[xpehh_params.map_pos] = None,
        min_ehh: xpehh_params.min_ehh = xpehh_params.min_ehh_default,
        max_gap: xpehh_params.max_gap = xpehh_params.max_gap_default,
        gap_scale: xpehh_params.gap_scale = xpehh_params.gap_scale_default,
        include_edges: xpehh_params.include_edges = True,
        use_threads: xpehh_params.use_threads = True,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = xpehh_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = xpehh_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        palette: xpehh_params.palette = xpehh_params.palette_default,
        title: Optional[gplt_params.title] = None,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        track_height: gplt_params.track_height = 170,
        genes_height: gplt_params.genes_height = gplt_params.genes_height_default,
        show: gplt_params.show = True,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        gene_labels: Optional[gplt_params.gene_labels] = None,
        gene_labelset: Optional[gplt_params.gene_labelset] = None,
    ) -> gplt_params.optional_figure:
        # gwss track
        fig1 = self.plot_xpehh_gwss_track(
            contig=contig,
            analysis=analysis,
            sample_sets=sample_sets,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_query_options=sample_query_options,
            window_size=window_size,
            percentiles=percentiles,
            palette=palette,
            filter_min_maf=filter_min_maf,
            map_pos=map_pos,
            min_ehh=min_ehh,
            max_gap=max_gap,
            gap_scale=gap_scale,
            include_edges=include_edges,
            use_threads=use_threads,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            title=title,
            sizing_mode=sizing_mode,
            width=width,
            height=track_height,
            show=False,
            x_range=None,
            output_backend=output_backend,
            chunks=chunks,
            inline_array=inline_array,
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
        )

        if show:  # pragma: no cover
            bokeh.plotting.show(fig)
            return None
        else:
            return fig
