from collections import Counter
from typing import Optional, Tuple, Dict, Mapping

import allel  # type: ignore
import numpy as np
from numpydoc_decorator import doc  # type: ignore
import bokeh.plotting

from .hap_data import AnophelesHapData
from ..util import hash_columns, check_types, CacheMiss
from . import base_params
from . import h12_params, gplt_params, hap_params


class AnophelesH12Analysis(
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

    def _h12_calibration(
        self,
        contig,
        analysis,
        sample_query,
        sample_sets,
        cohort_size,
        min_cohort_size,
        max_cohort_size,
        window_sizes,
        random_seed,
    ) -> Mapping[str, np.ndarray]:
        ds_haps = self.haplotypes(
            region=contig,
            sample_sets=sample_sets,
            sample_query=sample_query,
            analysis=analysis,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        gt = allel.GenotypeDaskArray(ds_haps["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes"):
            ht = gt.to_haplotypes().compute()

        calibration_runs: Dict[str, np.ndarray] = dict()
        for window_size in self._progress(window_sizes, desc="Compute H12"):
            h12 = allel.moving_statistic(ht, statistic=garud_h12, size=window_size)
            calibration_runs[str(window_size)] = h12

        return calibration_runs

    @check_types
    @doc(
        summary="Generate h12 GWSS calibration data for different window sizes.",
        returns="""
            A list of H12 calibration run arrays for each window size, containing
            values and percentiles.
        """,
    )
    def h12_calibration(
        self,
        contig: base_params.contig,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_query: Optional[base_params.sample_query] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        cohort_size: Optional[base_params.cohort_size] = h12_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = h12_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = h12_params.max_cohort_size_default,
        window_sizes: h12_params.window_sizes = h12_params.window_sizes_default,
        random_seed: base_params.random_seed = 42,
    ) -> Mapping[str, np.ndarray]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "h12_calibration_v1"

        params = dict(
            contig=contig,
            analysis=self._prep_phasing_analysis_param(analysis=analysis),
            window_sizes=window_sizes,
            sample_sets=self._prep_sample_sets_param(sample_sets=sample_sets),
            # N.B., do not be tempted to convert this sample query into integer
            # indices using _prep_sample_selection_params, because the indices
            # are different in the haplotype data.
            sample_query=sample_query,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        try:
            calibration_runs = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            calibration_runs = self._h12_calibration(**params)
            self.results_cache_set(name=name, params=params, results=calibration_runs)

        return calibration_runs

    @check_types
    @doc(
        summary="Plot h12 GWSS calibration data for different window sizes.",
        parameters=dict(
            title="Plot title.",
            show="If True, show the plot.",
        ),
    )
    def plot_h12_calibration(
        self,
        contig: base_params.contig,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_query: Optional[base_params.sample_query] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        cohort_size: Optional[base_params.cohort_size] = h12_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = h12_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = h12_params.max_cohort_size_default,
        window_sizes: h12_params.window_sizes = h12_params.window_sizes_default,
        random_seed: base_params.random_seed = 42,
        title: Optional[str] = None,
        show: bool = True,
    ) -> gplt_params.figure:
        # Get H12 values.
        calibration_runs = self.h12_calibration(
            contig=contig,
            analysis=analysis,
            sample_query=sample_query,
            sample_sets=sample_sets,
            window_sizes=window_sizes,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        # Compute summaries.
        q50 = [np.median(calibration_runs[str(window)]) for window in window_sizes]
        q25 = [
            np.percentile(calibration_runs[str(window)], 25) for window in window_sizes
        ]
        q75 = [
            np.percentile(calibration_runs[str(window)], 75) for window in window_sizes
        ]
        q05 = [
            np.percentile(calibration_runs[str(window)], 5) for window in window_sizes
        ]
        q95 = [
            np.percentile(calibration_runs[str(window)], 95) for window in window_sizes
        ]

        # Make plot.
        if title is None:
            title = sample_query
        fig = bokeh.plotting.figure(
            title=title,
            width=700,
            height=400,
            x_axis_type="log",
            x_range=bokeh.models.Range1d(window_sizes[0], window_sizes[-1]),
        )
        patch_x = list(window_sizes) + list(window_sizes[::-1])
        fig.patch(
            patch_x,
            q75 + q25[::-1],
            alpha=0.75,
            line_width=2,
            legend_label="25-75%",
        )
        fig.patch(
            patch_x,
            q95 + q05[::-1],
            alpha=0.5,
            line_width=2,
            legend_label="5-95%",
        )
        fig.line(
            window_sizes, q50, line_color="black", line_width=4, legend_label="median"
        )
        fig.scatter(
            window_sizes,
            q50,
            marker="circle",
            color="black",
            fill_color="black",
            size=8,
        )

        fig.xaxis.ticker = window_sizes
        if show:  # pragma: no cover
            bokeh.plotting.show(fig)
            return None
        else:
            return fig

    def _h12_gwss(
        self,
        contig,
        analysis,
        window_size,
        sample_sets,
        sample_query,
        cohort_size,
        min_cohort_size,
        max_cohort_size,
        random_seed,
    ):
        ds_haps = self.haplotypes(
            region=contig,
            analysis=analysis,
            sample_query=sample_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        gt = allel.GenotypeDaskArray(ds_haps["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes"):
            ht = gt.to_haplotypes().compute()

        with self._spinner(desc="Compute H12"):
            # Compute H12.
            h12 = allel.moving_statistic(ht, statistic=garud_h12, size=window_size)

            # Compute window midpoints.
            pos = ds_haps["variant_position"].values
            x = allel.moving_statistic(pos, statistic=np.mean, size=window_size)

        results = dict(x=x, h12=h12)

        return results

    @check_types
    @doc(
        summary="Run h12 genome-wide selection scan.",
        returns=dict(
            x="An array containing the window centre point genomic positions.",
            h12="An array with h12 statistic values for each window.",
        ),
    )
    def h12_gwss(
        self,
        contig: base_params.contig,
        window_size: h12_params.window_size,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_query: Optional[base_params.sample_query] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        cohort_size: Optional[base_params.cohort_size] = h12_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = h12_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = h12_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "h12_gwss_v1"

        params = dict(
            contig=contig,
            analysis=self._prep_phasing_analysis_param(analysis=analysis),
            window_size=window_size,
            sample_sets=self._prep_sample_sets_param(sample_sets=sample_sets),
            # N.B., do not be tempted to convert this sample query into integer
            # indices using _prep_sample_selection_params, because the indices
            # are different in the haplotype data.
            sample_query=sample_query,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._h12_gwss(**params)
            self.results_cache_set(name=name, params=params, results=results)

        x = results["x"]
        h12 = results["h12"]

        return x, h12

    @check_types
    @doc(
        summary="Plot h12 GWSS data.",
    )
    def plot_h12_gwss_track(
        self,
        contig: base_params.contig,
        window_size: h12_params.window_size,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        cohort_size: Optional[base_params.cohort_size] = h12_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = h12_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = h12_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        title: Optional[gplt_params.title] = None,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        height: gplt_params.height = 200,
        show: gplt_params.show = True,
        x_range: Optional[gplt_params.x_range] = None,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
    ) -> gplt_params.figure:
        # Compute H12.
        x, h12 = self.h12_gwss(
            contig=contig,
            analysis=analysis,
            window_size=window_size,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            sample_query=sample_query,
            sample_sets=sample_sets,
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
            title = sample_query
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

        # Plot H12.
        fig.scatter(
            x=x,
            y=h12,
            marker="circle",
            size=3,
            line_width=1,
            line_color="black",
            fill_color=None,
        )

        # Tidy up the plot.
        fig.yaxis.axis_label = "H12"
        fig.yaxis.ticker = [0, 1]
        self._bokeh_style_genome_xaxis(fig, contig)

        if show:  # pragma: no cover
            bokeh.plotting.show(fig)
            return None
        else:
            return fig

    @check_types
    @doc(
        summary="Plot h12 GWSS data.",
    )
    def plot_h12_gwss(
        self,
        contig: base_params.contig,
        window_size: h12_params.window_size,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        cohort_size: Optional[base_params.cohort_size] = h12_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = h12_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = h12_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        title: Optional[gplt_params.title] = None,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        track_height: gplt_params.track_height = 170,
        genes_height: gplt_params.genes_height = gplt_params.genes_height_default,
        show: gplt_params.show = True,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
    ) -> gplt_params.figure:
        # Plot GWSS track.
        fig1 = self.plot_h12_gwss_track(
            contig=contig,
            analysis=analysis,
            window_size=window_size,
            sample_sets=sample_sets,
            sample_query=sample_query,
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
        )

        # Combine plots into a single figure.
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


def haplotype_frequencies(h):
    """Compute haplotype frequencies, returning a dictionary that maps
    haplotype hash values to frequencies."""
    n = h.shape[1]
    hashes = hash_columns(np.asarray(h))
    counts = Counter(hashes)
    freqs = {key: count / n for key, count in counts.items()}
    return freqs


def garud_h12(ht):
    """Compute Garud's H12."""

    # Compute haplotype frequencies.
    frq_counter = haplotype_frequencies(ht)

    # Convert to array of sorted frequencies.
    f = np.sort(np.fromiter(frq_counter.values(), dtype=float))[::-1]

    # Compute H12.
    h12 = np.sum(f[:2]) ** 2 + np.sum(f[2:] ** 2)

    return h12
