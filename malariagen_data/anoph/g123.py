from collections import Counter
from typing import Optional, Tuple, Dict, Mapping

import allel  # type: ignore
import numpy as np
from numpydoc_decorator import doc  # type: ignore
import bokeh.plotting

from .snp_data import AnophelesSnpData
from .hap_data import AnophelesHapData
from ..util import hash_columns, check_types, CacheMiss
from . import base_params
from . import g123_params, gplt_params


class AnophelesG123Analysis(
    # Note that G123 analysis uses unphased genotype data for the main
    # calculation. However, when selecting which sites to use, it is an
    # option to use sites from a phasing analysis. This doesn't require
    # any haplotype data for the samples being analysed, it just requires
    # the data for the sites at which phasing is usually performed. The
    # rationale for using sites from a phasing analysis is that these are
    # a set of sites which were ascertained as polymorphic in a relatively
    # large panel of samples from different populations.
    AnophelesHapData,
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

    def _load_data_for_g123(
        self,
        *,
        contig,
        sites,
        site_mask,
        sample_sets,
        sample_query,
        min_cohort_size,
        max_cohort_size,
        random_seed,
        inline_array,
        chunks,
    ):
        ds_snps = self.snp_calls(
            region=contig,
            sample_query=sample_query,
            sample_sets=sample_sets,
            site_mask=site_mask,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            inline_array=inline_array,
            chunks=chunks,
        )

        with self._dask_progress(desc="Load genotypes"):
            gt = ds_snps["call_genotype"].data.compute()

        with self._dask_progress(desc="Load SNP positions"):
            pos = ds_snps["variant_position"].data.compute()

        if sites in self.phasing_analysis_ids:
            # Here we use sites from a phasing analysis. This is effectively
            # using a set of sites ascertained as polymorphic in whatever panel
            # of samples was used to set up the phasing analysis.
            with self._spinner("Subsetting to selected sites"):
                haplotype_pos = self.haplotype_sites(
                    region=contig,
                    analysis=sites,
                    field="POS",
                    inline_array=True,
                    chunks="native",
                ).compute()
                hap_site_mask = np.in1d(pos, haplotype_pos, assume_unique=True)
                pos = pos[hap_site_mask]
                gt = gt.compress(hap_site_mask, axis=0)

        elif sites == "segregating":
            # Here we use sites which are segregating within the samples
            # to be analysed. This is sometimes less preferable, because
            # a selective sweep can cause a deficit of segregating sites,
            # but windows for G123 calculation use a fixed number of SNPs.
            # This means that windows spanning a selective sweep can end
            # up covering a larger genome region, which in turn can weaken
            # the signal of a selective sweep. Hence it is generally better
            # to use sites ascertained as polymorphic in a different population
            # or panel of populations, for which using a phasing analysis is
            # a proxy.
            with self._spinner("Subsetting to segregating sites"):
                ac = allel.GenotypeArray(gt).count_alleles(max_allele=3)
                seg = ac.is_segregating()
                pos = pos[seg]
                gt = gt.compress(seg, axis=0)

        return gt, pos

    def _g123_gwss(
        self,
        *,
        contig,
        sites,
        site_mask,
        window_size,
        sample_sets,
        sample_query,
        min_cohort_size,
        max_cohort_size,
        random_seed,
        inline_array,
        chunks,
    ):
        gt, pos = self._load_data_for_g123(
            contig=contig,
            sites=sites,
            site_mask=site_mask,
            sample_sets=sample_sets,
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            inline_array=inline_array,
            chunks=chunks,
        )

        with self._spinner("Compute G123"):
            g123 = allel.moving_statistic(gt, statistic=garud_g123, size=window_size)
            x = allel.moving_statistic(pos, statistic=np.mean, size=window_size)

        results = dict(x=x, g123=g123)

        return results

    @check_types
    @doc(
        summary="Run a G123 genome-wide selection scan.",
        returns=dict(
            x="An array containing the window centre point genomic positions.",
            g123="An array with G123 statistic values for each window.",
        ),
    )
    def g123_gwss(
        self,
        contig: base_params.contig,
        window_size: g123_params.window_size,
        sites: g123_params.sites = base_params.DEFAULT,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = g123_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = g123_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "g123_gwss_v1"

        if sites == base_params.DEFAULT:
            assert self._default_phasing_analysis is not None
            sites = self._default_phasing_analysis
        valid_sites = self.phasing_analysis_ids + ("all", "segregating")
        if sites not in valid_sites:
            raise ValueError(
                f"Invalid value for `sites` parameter, must be one of {valid_sites}."
            )

        params = dict(
            contig=contig,
            sites=sites,
            site_mask=site_mask,
            window_size=window_size,
            sample_sets=self._prep_sample_sets_param(sample_sets=sample_sets),
            # N.B., do not be tempted to convert this sample query into integer
            # indices using _prep_sample_selection_params, because the indices
            # are different in the haplotype data.
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._g123_gwss(
                inline_array=inline_array, chunks=chunks, **params
            )
            self.results_cache_set(name=name, params=params, results=results)

        x = results["x"]
        g123 = results["g123"]

        return x, g123

    def _g123_calibration(
        self,
        *,
        contig,
        sites,
        site_mask,
        sample_query,
        sample_sets,
        min_cohort_size,
        max_cohort_size,
        window_sizes,
        random_seed,
        inline_array,
        chunks,
    ) -> Mapping[str, np.ndarray]:
        gt, _ = self._load_data_for_g123(
            contig=contig,
            sites=sites,
            site_mask=site_mask,
            sample_query=sample_query,
            sample_sets=sample_sets,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            inline_array=inline_array,
            chunks=chunks,
        )

        calibration_runs: Dict[str, np.ndarray] = dict()
        for window_size in self._progress(window_sizes, desc="Compute G123"):
            g123 = allel.moving_statistic(gt, statistic=garud_g123, size=window_size)
            calibration_runs[str(window_size)] = g123

        return calibration_runs

    @check_types
    @doc(
        summary="Generate G123 GWSS calibration data for different window sizes.",
        returns="""
            A list of G123 calibration run arrays for each window size, containing
            values and percentiles.
        """,
    )
    def g123_calibration(
        self,
        contig: base_params.contig,
        sites: g123_params.sites = base_params.DEFAULT,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        sample_query: Optional[base_params.sample_query] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = g123_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = g123_params.max_cohort_size_default,
        window_sizes: g123_params.window_sizes = g123_params.window_sizes_default,
        random_seed: base_params.random_seed = 42,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> Mapping[str, np.ndarray]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "g123_calibration_v1"

        params = dict(
            contig=contig,
            sites=sites,
            site_mask=self._prep_optional_site_mask_param(site_mask=site_mask),
            window_sizes=window_sizes,
            sample_sets=self._prep_sample_sets_param(sample_sets=sample_sets),
            # N.B., do not be tempted to convert this sample query into integer
            # indices using _prep_sample_selection_params, because the indices
            # are different in the haplotype data.
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        try:
            calibration_runs = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            calibration_runs = self._g123_calibration(
                inline_array=inline_array, chunks=chunks, **params
            )
            self.results_cache_set(name=name, params=params, results=calibration_runs)

        return calibration_runs

    @check_types
    @doc(
        summary="Plot G123 GWSS data.",
    )
    def plot_g123_gwss_track(
        self,
        contig: base_params.contig,
        window_size: g123_params.window_size,
        sites: g123_params.sites = base_params.DEFAULT,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = g123_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = g123_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        title: Optional[gplt_params.title] = None,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        height: gplt_params.height = 200,
        show: gplt_params.show = True,
        x_range: Optional[gplt_params.x_range] = None,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> gplt_params.figure:
        # compute G123
        x, g123 = self.g123_gwss(
            contig=contig,
            sites=sites,
            site_mask=site_mask,
            window_size=window_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            sample_query=sample_query,
            sample_sets=sample_sets,
            random_seed=random_seed,
            inline_array=inline_array,
            chunks=chunks,
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

        # plot G123
        fig.scatter(
            x=x,
            y=g123,
            size=3,
            marker="circle",
            line_width=1,
            line_color="black",
            fill_color=None,
        )

        # tidy up the plot
        fig.yaxis.axis_label = "G123"
        fig.yaxis.ticker = [0, 1]
        self._bokeh_style_genome_xaxis(fig, contig)

        if show:  # pragma: no cover
            bokeh.plotting.show(fig)
            return None
        else:
            return fig

    @check_types
    @doc(
        summary="Plot G123 GWSS data.",
    )
    def plot_g123_gwss(
        self,
        contig: base_params.contig,
        window_size: g123_params.window_size,
        sites: g123_params.sites = base_params.DEFAULT,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = g123_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = g123_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        title: Optional[gplt_params.title] = None,
        sizing_mode: gplt_params.sizing_mode = gplt_params.sizing_mode_default,
        width: gplt_params.width = gplt_params.width_default,
        track_height: gplt_params.track_height = 170,
        genes_height: gplt_params.genes_height = gplt_params.genes_height_default,
        show: gplt_params.show = True,
        output_backend: gplt_params.output_backend = gplt_params.output_backend_default,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> gplt_params.figure:
        # gwss track
        fig1 = self.plot_g123_gwss_track(
            contig=contig,
            sites=sites,
            site_mask=site_mask,
            window_size=window_size,
            sample_sets=sample_sets,
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            title=title,
            sizing_mode=sizing_mode,
            width=width,
            height=track_height,
            show=False,
            output_backend=output_backend,
            inline_array=inline_array,
            chunks=chunks,
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
        summary="Plot G123 GWSS calibration data for different window sizes.",
    )
    def plot_g123_calibration(
        self,
        contig: base_params.contig,
        sites: g123_params.sites,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        sample_query: Optional[base_params.sample_query] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = g123_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = g123_params.max_cohort_size_default,
        window_sizes: g123_params.window_sizes = g123_params.window_sizes_default,
        random_seed: base_params.random_seed = 42,
        title: Optional[gplt_params.title] = None,
        show: gplt_params.show = True,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> gplt_params.figure:
        # get g123 values
        calibration_runs = self.g123_calibration(
            contig=contig,
            sites=sites,
            site_mask=site_mask,
            sample_query=sample_query,
            sample_sets=sample_sets,
            window_sizes=window_sizes,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            inline_array=inline_array,
            chunks=chunks,
        )

        # compute summaries
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

        # make plot
        if title is None:
            title = sample_query
        fig = bokeh.plotting.figure(
            title=title,
            width=700,
            height=400,
            x_axis_type="log",
            x_range=bokeh.models.Range1d(window_sizes[0], window_sizes[-1]),
        )
        patch_x = tuple(window_sizes) + tuple(window_sizes)[::-1]
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


def diplotype_frequencies(gt):
    """Compute diplotype frequencies, returning a dictionary that maps
    diplotype hash values to frequencies."""

    # Here are some optimisations to speed up the computation
    # of diplotype hashes. First we combine the two int8 alleles
    # in each genotype call into a single int16.
    m = gt.shape[0]
    n = gt.shape[1]
    x = np.asarray(gt).view(np.int16).reshape((m, n))

    # Now call optimised hashing function.
    hashes = hash_columns(x)

    # Now compute counts and frequencies of distinct haplotypes.
    counts = Counter(hashes)
    freqs = {key: count / n for key, count in counts.items()}

    return freqs


def garud_g123(gt):
    """Compute Garud's G123."""

    # compute diplotype frequencies
    frq_counter = diplotype_frequencies(gt)

    # convert to array of sorted frequencies
    f = np.sort(np.fromiter(frq_counter.values(), dtype=float))[::-1]

    # compute G123
    g123 = np.sum(f[:3]) ** 2 + np.sum(f[3:] ** 2)

    # These other statistics are not currently needed, but leaving here
    # commented out for future reference...

    # compute G1
    # g1 = np.sum(f**2)

    # compute G12
    # g12 = np.sum(f[:2]) ** 2 + np.sum(f[2:] ** 2)  # type: ignore[index]

    # compute G2/G1
    # g2 = g1 - f[0] ** 2  # type: ignore[index]
    # g2_g1 = g2 / g1

    return g123
