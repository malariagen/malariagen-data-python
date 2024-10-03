from typing import Optional, Tuple

import allel  # type: ignore
import numpy as np
from numpydoc_decorator import doc  # type: ignore
import bokeh.plotting

from .hap_data import AnophelesHapData
from ..util import check_types, CacheMiss
from . import base_params
from . import h12_params, gplt_params, hap_params
from .h12 import haplotype_frequencies


class AnophelesH1XAnalysis(
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

    def _h1x_gwss_contig(
        self,
        contig,
        analysis,
        window_size,
        sample_sets,
        cohort1_query,
        cohort2_query,
        cohort_size,
        min_cohort_size,
        max_cohort_size,
        random_seed,
        chunks,
        inline_array,
    ):
        # Access haplotype datasets for each cohort.
        ds1 = self.haplotypes(
            region=contig,
            analysis=analysis,
            sample_query=cohort1_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            chunks=chunks,
            inline_array=inline_array,
        )
        ds2 = self.haplotypes(
            region=contig,
            analysis=analysis,
            sample_query=cohort2_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            chunks=chunks,
            inline_array=inline_array,
        )

        # Load data into memory.
        gt1 = allel.GenotypeDaskArray(ds1["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes for cohort 1"):
            ht1 = gt1.to_haplotypes().compute()
        gt2 = allel.GenotypeDaskArray(ds2["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes for cohort 2"):
            ht2 = gt2.to_haplotypes().compute()

        with self._spinner(desc="Compute H1X"):
            # Run H1X scan.
            h1x = moving_h1x(ht1, ht2, size=window_size)

            # Compute window midpoints.
            pos = ds1["variant_position"].values
            x = allel.moving_statistic(pos, statistic=np.mean, size=window_size)
            contigs = allel.moving_statistic(
                ds1["variant_contig"].values, statistic=np.median, size=window_size
            )

        results = dict(x=x, h1x=h1x, contigs=contigs)

        return results

    def _h1x_gwss(
        self,
        contig,
        analysis,
        window_size,
        sample_sets,
        cohort1_query,
        cohort2_query,
        cohort_size,
        min_cohort_size,
        max_cohort_size,
        random_seed,
        chunks,
        inline_array,
    ):
        results_tmp = self._h1x_gwss_contig(
            contig=contig,
            analysis=analysis,
            window_size=window_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            chunks=chunks,
            inline_array=inline_array,
        )

        results = dict(x=results_tmp["x"], h1x=results_tmp["h1x"])

        return results

    @check_types
    @doc(
        summary="""
            Run a H1X genome-wide scan to detect genome regions with
            shared selective sweeps between two cohorts.
        """,
        returns=dict(
            x="An array containing the window centre point genomic positions.",
            h1x="An array with H1X statistic values for each window.",
            contigs="An array with the contig for each window. The median is chosen for windows overlapping a change of contig.",
        ),
    )
    def h1x_gwss_contig(
        self,
        contig: base_params.contig,
        window_size: h12_params.window_size,
        cohort1_query: base_params.sample_query,
        cohort2_query: base_params.sample_query,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        cohort_size: Optional[base_params.cohort_size] = h12_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = h12_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = h12_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "h1x_gwss_contig_v1"

        params = dict(
            contig=contig,
            analysis=self._prep_phasing_analysis_param(analysis=analysis),
            window_size=window_size,
            # N.B., do not be tempted to convert these sample queries into integer
            # indices using _prep_sample_selection_params, because the indices
            # are different in the haplotype data.
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_sets=self._prep_sample_sets_param(sample_sets=sample_sets),
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._h1x_gwss_contig(
                chunks=chunks, inline_array=inline_array, **params
            )
            self.results_cache_set(name=name, params=params, results=results)

        x = results["x"]
        h1x = results["h1x"]
        contigs = results["contigs"]

        return x, h1x, contigs

    @check_types
    @doc(
        summary="""
            Run a H1X genome-wide scan to detect genome regions with
            shared selective sweeps between two cohorts.
        """,
        returns=dict(
            x="An array containing the window centre point genomic positions.",
            h1x="An array with H1X statistic values for each window.",
        ),
    )
    def h1x_gwss(
        self,
        contig: base_params.contig,
        window_size: h12_params.window_size,
        cohort1_query: base_params.sample_query,
        cohort2_query: base_params.sample_query,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        cohort_size: Optional[base_params.cohort_size] = h12_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = h12_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = h12_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "h1x_gwss_v1"

        params = dict(
            contig=contig,
            analysis=self._prep_phasing_analysis_param(analysis=analysis),
            window_size=window_size,
            # N.B., do not be tempted to convert these sample queries into integer
            # indices using _prep_sample_selection_params, because the indices
            # are different in the haplotype data.
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_sets=self._prep_sample_sets_param(sample_sets=sample_sets),
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._h1x_gwss(chunks=chunks, inline_array=inline_array, **params)
            self.results_cache_set(name=name, params=params, results=results)

        x = results["x"]
        h1x = results["h1x"]

        return x, h1x

    @check_types
    @doc(
        summary="""
            Run and plot a H1X genome-wide scan to detect genome regions
            with shared selective sweeps between two cohorts.
        """
    )
    def plot_h1x_gwss_track(
        self,
        contig: base_params.contig,
        window_size: h12_params.window_size,
        cohort1_query: base_params.cohort1_query,
        cohort2_query: base_params.cohort2_query,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
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
        circle_kwargs_param: Optional[gplt_params.circle_kwargs_param] = None,
        show: gplt_params.show = True,
        x_range: Optional[gplt_params.x_range] = None,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> gplt_params.figure:
        # Compute H1X.
        x, h1x, contigs = self.h1x_gwss_contig(
            contig=contig,
            analysis=analysis,
            window_size=window_size,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_sets=sample_sets,
            random_seed=random_seed,
            chunks=chunks,
            inline_array=inline_array,
        )

        circle_kwargs_param_dict: dict[int, gplt_params.circle_kwargs] = {}
        if circle_kwargs_param is None:
            circle_kwargs_param_dict = gplt_params.default_circle_kwargs_dict
        elif isinstance(circle_kwargs_param, list):
            if len(circle_kwargs_param) >= 5:
                circle_kwargs_param_dict = {
                    i: circle_kwargs_param[i] for i in range(0, 5)
                }
            else:
                circle_kwargs_param_dict = {
                    i: circle_kwargs_param[i]
                    for i in range(0, len(circle_kwargs_param))
                }
                circle_kwargs_param_dict.update(
                    {
                        i: gplt_params.default_circle_kwargs_dict[i]
                        for i in range(len(circle_kwargs_param), 5)
                    }
                )
        elif isinstance(circle_kwargs_param, dict):
            if isinstance(list(circle_kwargs_param.keys())[0], str):
                if list(circle_kwargs_param.keys())[0] in [
                    "2L",
                    "2R",
                    "3L",
                    "3R",
                    "X",
                    "2RL",
                    "3RL",
                ]:
                    for i in range(0, 5):
                        if i == 0:
                            if "2L" in circle_kwargs_param.keys():  # Ag3
                                circle_kwargs_param_dict[i] = circle_kwargs_param["2L"]
                            elif "2RL" in circle_kwargs_param.keys():  # Af1
                                circle_kwargs_param_dict[i] = circle_kwargs_param["2RL"]
                            else:
                                circle_kwargs_param_dict[
                                    i
                                ] = gplt_params.default_circle_kwargs_dict[i]
                        elif i == 1:
                            if "2R" in circle_kwargs_param.keys():  # Ag3
                                circle_kwargs_param_dict[i] = circle_kwargs_param["2R"]
                            elif "3RL" in circle_kwargs_param.keys():  # Af1
                                circle_kwargs_param_dict[i] = circle_kwargs_param["3RL"]
                            else:
                                circle_kwargs_param_dict[
                                    i
                                ] = gplt_params.default_circle_kwargs_dict[i]
                        elif i == 2:
                            if "3L" in circle_kwargs_param.keys():  # Ag3
                                circle_kwargs_param_dict[i] = circle_kwargs_param["3L"]
                            elif "X" in circle_kwargs_param.keys():  # Af1
                                circle_kwargs_param_dict[i] = circle_kwargs_param["X"]
                            else:
                                circle_kwargs_param_dict[
                                    i
                                ] = gplt_params.default_circle_kwargs_dict[i]
                        elif i == 3:
                            if "3R" in circle_kwargs_param.keys():  # Ag3
                                circle_kwargs_param_dict[i] = circle_kwargs_param["3R"]
                            else:
                                circle_kwargs_param_dict[
                                    i
                                ] = gplt_params.default_circle_kwargs_dict[i]
                        elif i == 4:
                            if (
                                "X" in circle_kwargs_param.keys()
                            ):  # Ag3. Will also get a value for Af1 but it will be ignored.
                                circle_kwargs_param_dict[i] = circle_kwargs_param["X"]
                            else:
                                circle_kwargs_param_dict[
                                    i
                                ] = gplt_params.default_circle_kwargs_dict[i]
                else:
                    print("circle_kwargs")
                    circle_kwargs_param_dict = {
                        i: circle_kwargs_param for i in range(0, 5)
                    }
            elif isinstance(list(circle_kwargs_param.keys())[0], int):
                circle_kwargs_param_dict = {}
                for i in range(0, 5):
                    if i in list(circle_kwargs_param.keys()):
                        circle_kwargs_param_dict[i] = circle_kwargs_param[i]
                    else:
                        circle_kwargs_param_dict[
                            i
                        ] = gplt_params.default_circle_kwargs_dict[i]
        else:
            circle_kwargs_param_dict = gplt_params.default_circle_kwargs_dict

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

        # Plot H1X.
        for s in set(contigs):
            idxs = contigs == s
            circle_kwargs_mutable = circle_kwargs_param_dict[s]
            fig.scatter(
                x=x[idxs],
                y=h1x[idxs],
                marker="circle",
                **circle_kwargs_mutable,
            )

        # Tidy up the plot.
        fig.yaxis.axis_label = "H1X"
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
            Run and plot a H1X genome-wide scan to detect genome regions
            with shared selective sweeps between two cohorts.
        """
    )
    def plot_h1x_gwss(
        self,
        contig: base_params.contig,
        window_size: h12_params.window_size,
        cohort1_query: base_params.cohort1_query,
        cohort2_query: base_params.cohort2_query,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
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
        track_height: gplt_params.track_height = 190,
        circle_kwargs_param: Optional[gplt_params.circle_kwargs_param] = None,
        genes_height: gplt_params.genes_height = gplt_params.genes_height_default,
        show: gplt_params.show = True,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> gplt_params.figure:
        # Plot GWSS track.
        fig1 = self.plot_h1x_gwss_track(
            contig=contig,
            analysis=analysis,
            window_size=window_size,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            title=title,
            sizing_mode=sizing_mode,
            width=width,
            height=track_height,
            circle_kwargs_param=circle_kwargs_param,
            show=False,
            output_backend=output_backend,
            chunks=chunks,
            inline_array=inline_array,
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


def haplotype_joint_frequencies(ha, hb):
    """Compute the joint frequency of haplotypes in two difference
    cohorts. Returns a dictionary mapping haplotype hash values to
    the product of frequencies in each cohort."""
    frqa = haplotype_frequencies(ha)
    frqb = haplotype_frequencies(hb)
    keys = set(frqa.keys()) | set(frqb.keys())
    joint_freqs = {key: frqa.get(key, 0) * frqb.get(key, 0) for key in keys}
    return joint_freqs


def h1x(ha, hb):
    """Compute H1X, the sum of joint haplotype frequencies between
    two cohorts, which is a summary statistic useful for detecting
    shared selective sweeps."""
    jf = haplotype_joint_frequencies(ha, hb)
    return np.sum(list(jf.values()))


def moving_h1x(ha, hb, size, start=0, stop=None, step=None):
    """Compute H1X in moving windows.

    Parameters
    ----------
    ha : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array for the first cohort.
    hb : array_like, int, shape (n_variants, n_haplotypes)
        Haplotype array for the second cohort.
    size : int
        The window size (number of variants).
    start : int, optional
        The index at which to start.
    stop : int, optional
        The index at which to stop.
    step : int, optional
        The number of variants between start positions of windows. If not
        given, defaults to the window size, i.e., non-overlapping windows.

    Returns
    -------
    h1x : ndarray, float, shape (n_windows,)
        H1X values (sum of squares of joint haplotype frequencies).
    """

    assert ha.ndim == hb.ndim == 2
    assert ha.shape[0] == hb.shape[0]

    # Construct moving windows.
    windows = allel.index_windows(ha, size, start, stop, step)

    # Compute statistics for each window.
    out = np.array([h1x(ha[i:j], hb[i:j]) for i, j in windows])

    return out
