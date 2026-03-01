from typing import Optional

import allel  # type: ignore
import bokeh.layouts
import bokeh.models
import bokeh.plotting
import numpy as np
import pandas as pd
from numpydoc_decorator import doc  # type: ignore

from .snp_data import AnophelesSnpData
from ..util import CacheMiss, Region, _check_types, _parse_single_region
from . import base_params, het_params, gplt_params


class AnophelesHetAnalysis(
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

    @staticmethod
    def _roh_hmm_predict(
        *,
        windows,
        counts,
        phet_roh,
        phet_nonroh,
        transition,
        window_size,
        sample_id,
        contig,
    ):
        # This implementation is based on scikit-allel, but modified to use
        # moving window computation of het counts.
        from allel.stats.misc import tabulate_state_blocks  # type: ignore
        from allel.stats.roh import _hmm_derive_transition_matrix  # type: ignore

        # Protopunica is pomegranate frozen at version 0.14.8, wich is compatible
        # with the code here. Also protopunica has binary wheels available from
        # PyPI and so installs much faster.
        from protopunica import HiddenMarkovModel, PoissonDistribution  # type: ignore

        # het probabilities
        het_px = np.concatenate([(phet_roh,), phet_nonroh])

        # start probabilities (all equal)
        start_prob = np.repeat(1 / het_px.size, het_px.size)

        # transition between underlying states
        transition_mx = _hmm_derive_transition_matrix(transition, het_px.size)

        # emission probability distribution
        dists = [PoissonDistribution(x * window_size) for x in het_px]

        # set up model
        # noinspection PyArgumentList
        model = HiddenMarkovModel.from_matrix(
            transition_probabilities=transition_mx,
            distributions=dists,
            starts=start_prob,
        )

        # predict hidden states
        prediction = np.array(model.predict(counts[:, None]))

        # tabulate runs of homozygosity (state 0)
        # noinspection PyTypeChecker
        df_blocks = tabulate_state_blocks(prediction, states=list(range(len(het_px))))
        df_roh = df_blocks[(df_blocks["state"] == 0)].reset_index(drop=True)

        # adapt the dataframe for ROH
        df_roh["sample_id"] = sample_id
        df_roh["contig"] = contig
        df_roh["roh_start"] = df_roh["start_ridx"].apply(lambda y: windows[y, 0])
        df_roh["roh_stop"] = df_roh["stop_lidx"].apply(lambda y: windows[y, 1])
        df_roh["roh_length"] = df_roh["roh_stop"] - df_roh["roh_start"]
        df_roh = df_roh.rename(columns={"is_marginal": "roh_is_marginal"})

        return df_roh[
            [
                "sample_id",
                "contig",
                "roh_start",
                "roh_stop",
                "roh_length",
                "roh_is_marginal",
            ]
        ]

    def _plot_heterozygosity_track(
        self,
        *,
        sample_id,
        sample_set,
        windows,
        counts,
        region: Region,
        window_size,
        y_max,
        sizing_mode,
        width,
        height,
        circle_kwargs,
        show,
        x_range,
        output_backend,
    ):
        debug = self._log.debug

        # pos axis
        window_pos = windows.mean(axis=1)

        # het axis
        window_het = counts / window_size

        # determine plotting limits
        if x_range is None:
            if region.start is not None:
                x_min = region.start
            else:
                x_min = 0
            if region.end is not None:
                x_max = region.end
            else:
                x_max = len(self.genome_sequence(region.contig))
            x_range = bokeh.models.Range1d(x_min, x_max, bounds="auto")

        debug("create a figure for plotting")
        xwheel_zoom = bokeh.models.WheelZoomTool(
            dimensions="width", maintain_focus=False
        )
        fig = bokeh.plotting.figure(
            title=f"{sample_id} ({sample_set})",
            tools=["xpan", "xzoom_in", "xzoom_out", xwheel_zoom, "reset", "save"],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            toolbar_location="above",
            x_range=x_range,
            y_range=(0, y_max),
            output_backend=output_backend,
        )

        debug("plot heterozygosity")
        data = pd.DataFrame(
            {
                "position": window_pos,
                "heterozygosity": window_het,
            }
        )
        if circle_kwargs is None:
            circle_kwargs = dict()
        circle_kwargs.setdefault("size", 4)
        circle_kwargs.setdefault("line_width", 0)
        fig.circle(x="position", y="heterozygosity", source=data, **circle_kwargs)

        debug("tidy up the plot")
        fig.yaxis.axis_label = "Heterozygosity (bp⁻¹)"
        self._bokeh_style_genome_xaxis(fig, region.contig)

        if show:  # pragma: no cover
            bokeh.plotting.show(fig)

        return fig

    @_check_types
    @doc(
        summary="Plot windowed heterozygosity for a single sample over a genome region.",
    )
    def plot_heterozygosity_track(
        self,
        sample: base_params.sample,
        region: base_params.region,
        window_size: het_params.window_size = het_params.window_size_default,
        y_max: het_params.y_max = het_params.y_max_default,
        circle_kwargs: Optional[gplt_params.circle_kwargs] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        sample_set: Optional[base_params.sample_set] = None,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        height: gplt_params.height = 200,
        show: gplt_params.show = True,
        x_range: Optional[gplt_params.x_range] = None,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> gplt_params.optional_figure:
        debug = self._log.debug

        # Normalise parameters.
        region_prepped: Region = _parse_single_region(self, region)
        del region

        debug("compute windowed heterozygosity")
        sample_id, sample_set, windows, counts = self._sample_count_het(
            sample=sample,
            region=region_prepped,
            site_mask=site_mask,
            window_size=window_size,
            sample_set=sample_set,
            chunks=chunks,
            inline_array=inline_array,
        )

        debug("plot heterozygosity")
        fig = self._plot_heterozygosity_track(
            sample_id=sample_id,
            sample_set=sample_set,
            windows=windows,
            counts=counts,
            region=region_prepped,
            window_size=window_size,
            y_max=y_max,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            circle_kwargs=circle_kwargs,
            show=show,
            x_range=x_range,
            output_backend=output_backend,
        )

        if show:  # pragma: no cover
            bokeh.plotting.show(fig)
            return None
        else:
            return fig

    @_check_types
    @doc(
        summary="Plot windowed heterozygosity for a single sample over a genome region.",
    )
    def plot_heterozygosity(
        self,
        sample: base_params.samples,
        region: base_params.region,
        window_size: het_params.window_size = het_params.window_size_default,
        y_max: het_params.y_max = het_params.y_max_default,
        circle_kwargs: Optional[gplt_params.circle_kwargs] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        sample_set: Optional[base_params.sample_set] = None,
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
        debug = self._log.debug

        # normalise to support multiple samples
        if isinstance(sample, (list, tuple)):
            samples = sample
        else:
            samples = [sample]

        debug("plot first sample track")
        fig1 = self.plot_heterozygosity_track(
            sample=samples[0],
            sample_set=sample_set,
            region=region,
            site_mask=site_mask,
            window_size=window_size,
            y_max=y_max,
            sizing_mode=sizing_mode,
            width=width,
            height=track_height,
            circle_kwargs=circle_kwargs,
            show=False,
            output_backend=output_backend,
            chunks=chunks,
            inline_array=inline_array,
        )
        fig1.xaxis.visible = False
        figs = [fig1]

        debug("plot remaining sample tracks")
        for sample in samples[1:]:
            fig_het = self.plot_heterozygosity_track(
                sample=sample,
                sample_set=sample_set,
                region=region,
                site_mask=site_mask,
                window_size=window_size,
                y_max=y_max,
                sizing_mode=sizing_mode,
                width=width,
                height=track_height,
                circle_kwargs=circle_kwargs,
                show=False,
                x_range=fig1.x_range,
                output_backend=output_backend,
                chunks=chunks,
                inline_array=inline_array,
            )
            fig_het.xaxis.visible = False
            figs.append(fig_het)

        debug("plot genes track")
        fig_genes = self.plot_genes(
            region=region,
            sizing_mode=sizing_mode,
            width=width,
            height=genes_height,
            x_range=fig1.x_range,
            show=False,
            output_backend=output_backend,
            gene_labels=gene_labels,
            gene_labelset=gene_labelset,
        )
        figs.append(fig_genes)

        debug("combine plots into a single figure")
        fig_all = bokeh.layouts.gridplot(
            figs,
            ncols=1,
            toolbar_location="above",
            merge_tools=True,
            sizing_mode=sizing_mode,
        )

        if show:  # pragma: no cover
            bokeh.plotting.show(fig_all)
            return None
        else:
            return fig_all

    def _sample_count_het(
        self,
        sample: base_params.sample,
        region: Region,
        site_mask: Optional[base_params.site_mask],
        window_size: het_params.window_size,
        sample_set: Optional[base_params.sample_set],
        chunks: base_params.chunks,
        inline_array: base_params.inline_array,
    ):
        debug = self._log.debug

        debug("access sample metadata, look up sample")
        sample_rec = self.lookup_sample(sample=sample, sample_set=sample_set)
        sample_id = sample_rec.name  # sample_id
        sample_set = sample_rec["sample_set"]

        debug("access SNPs, select data for sample")
        ds_snps = self.snp_calls(
            region=region,
            sample_sets=sample_set,
            site_mask=site_mask,
            chunks=chunks,
            inline_array=inline_array,
        )
        ds_snps_sample = ds_snps.set_index(samples="sample_id").sel(samples=sample_id)

        # snp positions
        pos = ds_snps_sample["variant_position"].values

        # access genotypes
        gt = allel.GenotypeDaskVector(ds_snps_sample["call_genotype"].data)

        # compute het
        with self._dask_progress(desc="Compute heterozygous genotypes"):
            is_het = gt.is_het().compute()

        # guard against window_size exceeding available sites
        if pos.shape[0] < window_size:
            raise ValueError(
                f"Not enough sites ({pos.shape[0]}) for window size "
                f"({window_size}). Please reduce the window size or "
                f"use different site selection criteria."
            )

        # compute window coordinates
        windows = allel.moving_statistic(
            values=pos,
            statistic=lambda x: [x[0], x[-1]],
            size=window_size,
        )

        # compute windowed heterozygosity
        counts = allel.moving_statistic(
            values=is_het,
            statistic=np.sum,
            size=window_size,
        )

        return sample_id, sample_set, windows, counts

    @property
    def _roh_hmm_cache_name(self):
        return "roh_hmm_v1"

    @_check_types
    @doc(
        summary="Infer runs of homozygosity for a single sample over a genome region.",
    )
    def roh_hmm(
        self,
        sample: base_params.sample,
        region: base_params.region,
        window_size: het_params.window_size = het_params.window_size_default,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        sample_set: Optional[base_params.sample_set] = None,
        phet_roh: het_params.phet_roh = het_params.phet_roh_default,
        phet_nonroh: het_params.phet_nonroh = het_params.phet_nonroh_default,
        transition: het_params.transition = het_params.transition_default,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> het_params.df_roh:
        debug = self._log.debug

        resolved_region: Region = _parse_single_region(self, region)

        name = self._roh_hmm_cache_name

        params = dict(
            sample=sample,
            region=region,
            window_size=window_size,
            site_mask=site_mask,
            sample_set=sample_set,
            phet_roh=phet_roh,
            phet_nonroh=phet_nonroh,
            transition=transition,
            chunks=chunks,
            inline_array=inline_array,
        )

        del region

        try:
            # Load cached numeric data, adding str / obj data again.
            results = self.results_cache_get(name=name, params=params)

            # Reconstruct dataframe
            df_roh = pd.DataFrame(
                {
                    "roh_start": results["roh_start"],
                    "roh_stop": results["roh_stop"],
                    "roh_length": results["roh_length"],
                    "roh_is_marginal": results["roh_is_marginal"],
                }
            )

            df_roh["sample_id"] = sample
            df_roh["contig"] = resolved_region.contig

        except CacheMiss:
            debug("compute windowed heterozygosity")
            sample_id, sample_set, windows, counts = self._sample_count_het(
                sample=sample,
                region=resolved_region,
                site_mask=site_mask,
                window_size=window_size,
                sample_set=sample_set,
                chunks=chunks,
                inline_array=inline_array,
            )

            debug("compute runs of homozygosity")
            df_roh = self._roh_hmm_predict(
                windows=windows,
                counts=counts,
                phet_roh=phet_roh,
                phet_nonroh=phet_nonroh,
                transition=transition,
                window_size=window_size,
                sample_id=sample_id,
                contig=resolved_region.contig,
            )

            # Specify numeric columns to save (saving obj - sample ID and contig - breaks the save.
            columns_to_save = [
                "roh_start",
                "roh_stop",
                "roh_length",
                "roh_is_marginal",
            ]

            self.results_cache_set(
                name=name,
                params=params,
                results={col: df_roh[col].to_numpy() for col in columns_to_save},
            )

        return df_roh

    @_check_types
    @doc(
        summary="Plot a runs of homozygosity track.",
    )
    def plot_roh_track(
        self,
        df_roh: het_params.df_roh,
        region: base_params.region,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        height: gplt_params.height = 80,
        show: gplt_params.show = True,
        x_range: Optional[gplt_params.x_range] = None,
        title: Optional[gplt_params.title] = None,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
    ) -> gplt_params.optional_figure:
        debug = self._log.debug

        debug("handle region parameter - this determines the genome region to plot")
        resolved_region: Region = _parse_single_region(self, region)
        del region
        contig = resolved_region.contig
        start = resolved_region.start
        end = resolved_region.end
        if start is None:
            start = 0
        if end is None:
            end = len(self.genome_sequence(contig))

        debug("define x axis range")
        if x_range is None:
            x_range = bokeh.models.Range1d(start, end, bounds="auto")

        debug(
            "we're going to plot each gene as a rectangle, so add some additional columns"
        )
        data = df_roh.copy()
        data["bottom"] = 0.2
        data["top"] = 0.8

        debug("make a figure")
        xwheel_zoom = bokeh.models.WheelZoomTool(
            dimensions="width", maintain_focus=False
        )
        fig = bokeh.plotting.figure(
            title=title,
            sizing_mode=sizing_mode,
            width=width,
            height=height,
            tools=[
                "xpan",
                "xzoom_in",
                "xzoom_out",
                xwheel_zoom,
                "reset",
                "tap",
                "hover",
                "save",
            ],
            active_scroll=xwheel_zoom,
            active_drag="xpan",
            x_range=x_range,
            y_range=bokeh.models.Range1d(0, 1),
            output_backend=output_backend,
        )

        debug("now plot the ROH as rectangles")
        fig.quad(
            bottom="bottom",
            top="top",
            left="roh_start",
            right="roh_stop",
            source=data,
            line_width=1,
            fill_alpha=0.5,
        )

        debug("tidy up the plot")
        fig.ygrid.visible = False
        fig.yaxis.ticker = []
        fig.yaxis.axis_label = "RoH"
        self._bokeh_style_genome_xaxis(fig, resolved_region.contig)

        if show:  # pragma: no cover
            bokeh.plotting.show(fig)
            return None
        else:
            return fig

    @_check_types
    @doc(
        summary="""
            Plot windowed heterozygosity and inferred runs of homozygosity for a
            single sample over a genome region.
        """,
    )
    def plot_roh(
        self,
        sample: base_params.sample,
        region: base_params.region,
        window_size: het_params.window_size = het_params.window_size_default,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        sample_set: Optional[base_params.sample_set] = None,
        phet_roh: het_params.phet_roh = het_params.phet_roh_default,
        phet_nonroh: het_params.phet_nonroh = het_params.phet_nonroh_default,
        transition: het_params.transition = het_params.transition_default,
        y_max: het_params.y_max = het_params.y_max_default,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        heterozygosity_height: gplt_params.height = 170,
        roh_height: gplt_params.height = 40,
        genes_height: gplt_params.genes_height = gplt_params.genes_height_default,
        circle_kwargs: Optional[gplt_params.circle_kwargs] = None,
        show: gplt_params.show = True,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        gene_labels: Optional[gplt_params.gene_labels] = None,
        gene_labelset: Optional[gplt_params.gene_labelset] = None,
    ) -> gplt_params.optional_figure:
        debug = self._log.debug

        resolved_region: Region = _parse_single_region(self, region)
        del region

        debug("compute windowed heterozygosity")
        sample_id, sample_set, windows, counts = self._sample_count_het(
            sample=sample,
            region=resolved_region,
            site_mask=site_mask,
            window_size=window_size,
            sample_set=sample_set,
            chunks=chunks,
            inline_array=inline_array,
        )

        debug("plot_heterozygosity track")
        fig_het = self._plot_heterozygosity_track(
            sample_id=sample_id,
            sample_set=sample_set,
            windows=windows,
            counts=counts,
            region=resolved_region,
            window_size=window_size,
            y_max=y_max,
            sizing_mode=sizing_mode,
            width=width,
            height=heterozygosity_height,
            circle_kwargs=circle_kwargs,
            show=False,
            x_range=None,
            output_backend=output_backend,
        )
        fig_het.xaxis.visible = False
        figs = [fig_het]

        debug("compute runs of homozygosity")
        df_roh = self._roh_hmm_predict(
            windows=windows,
            counts=counts,
            phet_roh=phet_roh,
            phet_nonroh=phet_nonroh,
            transition=transition,
            window_size=window_size,
            sample_id=sample_id,
            contig=resolved_region.contig,
        )

        debug("plot roh track")
        fig_roh = self.plot_roh_track(
            df_roh,
            region=resolved_region,
            sizing_mode=sizing_mode,
            width=width,
            height=roh_height,
            show=False,
            x_range=fig_het.x_range,
            output_backend=output_backend,
        )
        fig_roh.xaxis.visible = False
        figs.append(fig_roh)

        debug("plot genes track")
        fig_genes = self.plot_genes(
            region=resolved_region,
            sizing_mode=sizing_mode,
            width=width,
            height=genes_height,
            x_range=fig_het.x_range,
            show=False,
            output_backend=output_backend,
            gene_labels=gene_labels,
            gene_labelset=gene_labelset,
        )
        figs.append(fig_genes)

        debug("combine plots into a single figure")
        fig_all = bokeh.layouts.gridplot(
            figs,
            ncols=1,
            toolbar_location="above",
            merge_tools=True,
            sizing_mode=sizing_mode,
        )

        if show:  # pragma: no cover
            bokeh.plotting.show(fig_all)
            return None
        else:
            return fig_all
