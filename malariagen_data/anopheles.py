from abc import abstractmethod
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import allel  # type: ignore
import bokeh.layouts
import bokeh.models
import bokeh.palettes
import bokeh.plotting
import numpy as np
from numpydoc_decorator import doc  # type: ignore

from .anoph import (
    aim_params,
    base_params,
    dash_params,
    gplt_params,
    hapnet_params,
    ihs_params,
    plotly_params,
    xpehh_params,
)
from .anoph.aim_data import AnophelesAimData
from .anoph.base import AnophelesBase
from .anoph.describe import AnophelesDescribe
from .anoph.dipclust import AnophelesDipClustAnalysis
from .anoph.distance import AnophelesDistanceAnalysis
from .anoph.diversity import AnophelesDiversityAnalysis  # new commit
from .anoph.fst import AnophelesFstAnalysis
from .anoph.g123 import AnophelesG123Analysis
from .anoph.genome_features import AnophelesGenomeFeaturesData
from .anoph.genome_sequence import AnophelesGenomeSequenceData
from .anoph.h1x import AnophelesH1XAnalysis
from .anoph.h12 import AnophelesH12Analysis
from .anoph.hap_data import AnophelesHapData, hap_params
from .anoph.hap_frq import AnophelesHapFrequencyAnalysis
from .anoph.hapclust import AnophelesHapClustAnalysis
from .anoph.heterozygosity import AnophelesHetAnalysis
from .anoph.igv import AnophelesIgv
from .anoph.karyotype import AnophelesKaryotypeAnalysis
from .anoph.pca import AnophelesPca
from .anoph.phenotypes import AnophelesPhenotypeData
from .anoph.sample_metadata import AnophelesSampleMetadata
from .anoph.snp_data import AnophelesSnpData
from .anoph.to_plink import PlinkConverter
from .mjn import _median_joining_network, _mjn_graph
from .util import Region  # noqa: F401 (re-exported via __init__.py)
from .util import CacheMiss, _check_types, _plotly_discrete_legend

# N.B., we are in the process of breaking up the AnophelesDataResource
# class into multiple parent classes like AnophelesGenomeSequenceData
# and AnophelesBase. This is work in progress, and further PRs are
# expected to factor out functions defined here in to separate classes.
# For more information, see:
#
# https://github.com/malariagen/malariagen-data-python/issues/366
#
# N.B., we are making use of multiple inheritance here, using co-operative
# classes. Because of the way that multiple inheritance works in Python,
# it is important that these parent classes are provided in a particular
# order. Otherwise the linearization of parent classes will fail. For
# more information about superclass linearization and method resolution
# order in Python, the following links may be useful.
#
# https://en.wikipedia.org/wiki/C3_linearization
# https://rhettinger.wordpress.com/2011/05/26/super-considered-super/


# work around pycharm failing to recognise that doc() is callable
# noinspection PyCallingNonCallable
class AnophelesDataResource(
    AnophelesDipClustAnalysis,
    AnophelesHapClustAnalysis,
    AnophelesH1XAnalysis,
    AnophelesH12Analysis,
    AnophelesG123Analysis,
    AnophelesFstAnalysis,
    AnophelesHetAnalysis,
    AnophelesHapFrequencyAnalysis,
    AnophelesDistanceAnalysis,
    AnophelesDiversityAnalysis,
    AnophelesPca,
    PlinkConverter,
    AnophelesIgv,
    AnophelesKaryotypeAnalysis,
    AnophelesAimData,
    AnophelesHapData,
    AnophelesSnpData,
    AnophelesSampleMetadata,
    AnophelesGenomeFeaturesData,
    AnophelesGenomeSequenceData,
    AnophelesDescribe,
    AnophelesBase,
    AnophelesPhenotypeData,
):
    """Anopheles data resources."""

    def __init__(
        self,
        url,
        public_url,
        config_path,
        cohorts_analysis: Optional[str],
        aim_analysis: Optional[str],
        aim_metadata_dtype: Optional[Mapping[str, Any]],
        aim_ids: Optional[aim_params.aim_ids],
        aim_palettes: Optional[aim_params.aim_palettes],
        site_filters_analysis: Optional[str],
        discordant_read_calls_analysis: Optional[str],
        default_site_mask: Optional[str],
        default_phasing_analysis: Optional[str],
        default_coverage_calls_analysis: Optional[str],
        bokeh_output_notebook: bool,
        results_cache: Optional[str],
        log,
        debug,
        show_progress,
        check_location,
        pre,
        gcs_default_url: Optional[str],
        gcs_region_urls: Mapping[str, str],
        major_version_number: int,
        major_version_path: str,
        gff_gene_type: str,
        gff_gene_name_attribute: str,
        gff_default_attributes: Tuple[str, ...],
        tqdm_class,
        storage_options: Mapping,
        taxon_colors: Optional[Mapping[str, str]] = None,
        aim_species_colors: Optional[Mapping[str, str]] = None,
        virtual_contigs: Optional[Mapping[str, Sequence[str]]] = None,
        gene_names: Optional[Mapping[str, str]] = None,
        inversion_tag_path: Optional[str] = None,
        unrestricted_use_only: Optional[bool] = None,
        surveillance_use_only: Optional[bool] = None,
    ):
        super().__init__(
            url=url,
            public_url=public_url,
            config_path=config_path,
            bokeh_output_notebook=bokeh_output_notebook,
            log=log,
            debug=debug,
            show_progress=show_progress,
            check_location=check_location,
            pre=pre,
            gcs_default_url=gcs_default_url,
            gcs_region_urls=gcs_region_urls,
            major_version_number=major_version_number,
            major_version_path=major_version_path,
            storage_options=storage_options,
            gff_gene_type=gff_gene_type,
            gff_gene_name_attribute=gff_gene_name_attribute,
            gff_default_attributes=gff_default_attributes,
            cohorts_analysis=cohorts_analysis,
            aim_analysis=aim_analysis,
            aim_metadata_dtype=aim_metadata_dtype,
            aim_ids=aim_ids,
            aim_palettes=aim_palettes,
            site_filters_analysis=site_filters_analysis,
            discordant_read_calls_analysis=discordant_read_calls_analysis,
            default_site_mask=default_site_mask,
            default_phasing_analysis=default_phasing_analysis,
            default_coverage_calls_analysis=default_coverage_calls_analysis,
            results_cache=results_cache,
            tqdm_class=tqdm_class,
            taxon_colors=taxon_colors,
            aim_species_colors=aim_species_colors,
            virtual_contigs=virtual_contigs,
            gene_names=gene_names,
            inversion_tag_path=inversion_tag_path,
            unrestricted_use_only=unrestricted_use_only,
            surveillance_use_only=surveillance_use_only,
        )

    @property
    @abstractmethod
    def _xpehh_gwss_cache_name(self):
        raise NotImplementedError("Must override _xpehh_gwss_cache_name")

    @property
    @abstractmethod
    def _ihs_gwss_cache_name(self):
        raise NotImplementedError("Must override _ihs_gwss_cache_name")

    @staticmethod
    def _make_gene_cnv_label(gene_id, gene_name, cnv_type):
        label = gene_id
        if isinstance(gene_name, str):
            label += f" ({gene_name})"
        label += f" {cnv_type}"
        return label

    @_check_types
    @doc(
        summary="Run iHS GWSS.",
        returns=dict(
            x="An array containing the window centre point genomic positions.",
            ihs="An array with iHS statistic values for each window.",
        ),
    )
    def ihs_gwss(
        self,
        contig: base_params.contig,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        window_size: ihs_params.window_size = ihs_params.window_size_default,
        percentiles: ihs_params.percentiles = ihs_params.percentiles_default,
        standardize: ihs_params.standardize = True,
        standardization_bins: Optional[ihs_params.standardization_bins] = None,
        standardization_n_bins: ihs_params.standardization_n_bins = ihs_params.standardization_n_bins_default,
        standardization_diagnostics: ihs_params.standardization_diagnostics = False,
        filter_min_maf: ihs_params.filter_min_maf = ihs_params.filter_min_maf_default,
        compute_min_maf: ihs_params.compute_min_maf = ihs_params.compute_min_maf_default,
        min_ehh: ihs_params.min_ehh = ihs_params.min_ehh_default,
        max_gap: ihs_params.max_gap = ihs_params.max_gap_default,
        gap_scale: ihs_params.gap_scale = ihs_params.gap_scale_default,
        include_edges: ihs_params.include_edges = True,
        use_threads: ihs_params.use_threads = True,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = ihs_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = ihs_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data
        name = self._ihs_gwss_cache_name

        params = dict(
            contig=contig,
            analysis=self._prep_phasing_analysis_param(analysis=analysis),
            window_size=window_size,
            percentiles=percentiles,
            standardize=standardize,
            standardization_bins=standardization_bins,
            standardization_n_bins=standardization_n_bins,
            standardization_diagnostics=standardization_diagnostics,
            filter_min_maf=filter_min_maf,
            compute_min_maf=compute_min_maf,
            min_ehh=min_ehh,
            include_edges=include_edges,
            max_gap=max_gap,
            gap_scale=gap_scale,
            use_threads=use_threads,
            sample_sets=self._prep_sample_sets_param(sample_sets=sample_sets),
            # N.B., do not be tempted to convert this sample query into integer
            # indices using _prep_sample_selection_params, because the indices
            # are different in the haplotype data.
            sample_query=self._prep_sample_query_param(sample_query=sample_query),
            sample_query_options=sample_query_options,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._ihs_gwss(chunks=chunks, inline_array=inline_array, **params)
            self.results_cache_set(name=name, params=params, results=results)

        x = results["x"]
        ihs = results["ihs"]

        return x, ihs

    def _ihs_gwss(
        self,
        *,
        contig,
        analysis,
        sample_sets,
        sample_query,
        sample_query_options,
        window_size,
        percentiles,
        standardize,
        standardization_bins,
        standardization_n_bins,
        standardization_diagnostics,
        filter_min_maf,
        compute_min_maf,
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
        ds_haps = self.haplotypes(
            region=contig,
            analysis=analysis,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            sample_sets=sample_sets,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            chunks=chunks,
            inline_array=inline_array,
        )

        gt = allel.GenotypeDaskArray(ds_haps["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes"):
            ht = gt.to_haplotypes().compute()

        with self._spinner(desc="Compute IHS"):
            ac = ht.count_alleles(max_allele=1)
            pos = ds_haps["variant_position"].values

            if filter_min_maf > 0:
                af = ac.to_frequencies()
                maf = np.min(af, axis=1)
                maf_filter = maf > filter_min_maf
                ht = ht.compress(maf_filter, axis=0)
                pos = pos[maf_filter]
                ac = ac[maf_filter]

            # compute iHS
            ihs = allel.ihs(
                h=ht,
                pos=pos,
                min_maf=compute_min_maf,
                min_ehh=min_ehh,
                include_edges=include_edges,
                max_gap=max_gap,
                gap_scale=gap_scale,
                use_threads=use_threads,
            )

            # remove any NaNs
            na_mask = ~np.isnan(ihs)
            ihs = ihs[na_mask]
            pos = pos[na_mask]
            ac = ac[na_mask]

            # take absolute value
            ihs = np.fabs(ihs)

            if standardize:
                ihs, _ = allel.standardize_by_allele_count(
                    score=ihs,
                    aac=ac[:, 1],
                    bins=standardization_bins,
                    n_bins=standardization_n_bins,
                    diagnostics=standardization_diagnostics,
                )

            if window_size:
                ihs = allel.moving_statistic(
                    ihs, statistic=np.percentile, size=window_size, q=percentiles
                )
                pos = allel.moving_statistic(pos, statistic=np.mean, size=window_size)

        results = dict(x=pos, ihs=ihs)

        return results

    @_check_types
    @doc(
        summary="Run and plot iHS GWSS data.",
    )
    def plot_ihs_gwss_track(
        self,
        contig: base_params.contig,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        window_size: ihs_params.window_size = ihs_params.window_size_default,
        percentiles: ihs_params.percentiles = ihs_params.percentiles_default,
        standardize: ihs_params.standardize = True,
        standardization_bins: Optional[ihs_params.standardization_bins] = None,
        standardization_n_bins: ihs_params.standardization_n_bins = ihs_params.standardization_n_bins_default,
        standardization_diagnostics: ihs_params.standardization_diagnostics = False,
        filter_min_maf: ihs_params.filter_min_maf = ihs_params.filter_min_maf_default,
        compute_min_maf: ihs_params.compute_min_maf = ihs_params.compute_min_maf_default,
        min_ehh: ihs_params.min_ehh = ihs_params.min_ehh_default,
        max_gap: ihs_params.max_gap = ihs_params.max_gap_default,
        gap_scale: ihs_params.gap_scale = ihs_params.gap_scale_default,
        include_edges: ihs_params.include_edges = True,
        use_threads: ihs_params.use_threads = True,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = ihs_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = ihs_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        palette: ihs_params.palette = ihs_params.palette_default,
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
        # compute ihs
        x, ihs = self.ihs_gwss(
            contig=contig,
            analysis=analysis,
            window_size=window_size,
            percentiles=percentiles,
            standardize=standardize,
            standardization_bins=standardization_bins,
            standardization_n_bins=standardization_n_bins,
            standardization_diagnostics=standardization_diagnostics,
            filter_min_maf=filter_min_maf,
            compute_min_maf=compute_min_maf,
            min_ehh=min_ehh,
            max_gap=max_gap,
            gap_scale=gap_scale,
            include_edges=include_edges,
            use_threads=use_threads,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            sample_sets=sample_sets,
            random_seed=random_seed,
            chunks=chunks,
            inline_array=inline_array,
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
            output_backend=output_backend,
        )

        if window_size:
            if isinstance(percentiles, int):
                percentiles = (percentiles,)
            # Ensure percentiles are sorted so that colors make sense.
            percentiles = tuple(sorted(percentiles))

        # add an empty dimension to ihs array if 1D
        ihs = np.reshape(ihs, (ihs.shape[0], -1))

        # select the base color palette to work from
        base_palette = bokeh.palettes.all_palettes[palette][8]

        # keep only enough colours to plot the IHS tracks
        bokeh_palette = base_palette[: ihs.shape[1]]

        # reverse the colors so darkest is last
        bokeh_palette = bokeh_palette[::-1]

        # plot IHS tracks
        for i in range(ihs.shape[1]):
            ihs_perc = ihs[:, i]
            color = bokeh_palette[i]

            # plot ihs
            fig.circle(
                x=x,
                y=ihs_perc,
                size=4,
                line_width=0,
                line_color=color,
                fill_color=color,
            )

        # tidy up the plot
        fig.yaxis.axis_label = "ihs"
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

    @doc(
        summary="Run and plot iHS GWSS data.",
    )
    def plot_ihs_gwss(
        self,
        contig: base_params.contig,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        window_size: ihs_params.window_size = ihs_params.window_size_default,
        percentiles: ihs_params.percentiles = ihs_params.percentiles_default,
        standardize: ihs_params.standardize = True,
        standardization_bins: Optional[ihs_params.standardization_bins] = None,
        standardization_n_bins: ihs_params.standardization_n_bins = ihs_params.standardization_n_bins_default,
        standardization_diagnostics: ihs_params.standardization_diagnostics = False,
        filter_min_maf: ihs_params.filter_min_maf = ihs_params.filter_min_maf_default,
        compute_min_maf: ihs_params.compute_min_maf = ihs_params.compute_min_maf_default,
        min_ehh: ihs_params.min_ehh = ihs_params.min_ehh_default,
        max_gap: ihs_params.max_gap = ihs_params.max_gap_default,
        gap_scale: ihs_params.gap_scale = ihs_params.gap_scale_default,
        include_edges: ihs_params.include_edges = True,
        use_threads: ihs_params.use_threads = True,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = ihs_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = ihs_params.max_cohort_size_default,
        random_seed: base_params.random_seed = 42,
        palette: ihs_params.palette = ihs_params.palette_default,
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
        fig1 = self.plot_ihs_gwss_track(
            contig=contig,
            analysis=analysis,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            window_size=window_size,
            percentiles=percentiles,
            palette=palette,
            standardize=standardize,
            standardization_bins=standardization_bins,
            standardization_n_bins=standardization_n_bins,
            standardization_diagnostics=standardization_diagnostics,
            filter_min_maf=filter_min_maf,
            compute_min_maf=compute_min_maf,
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
        name = self._xpehh_gwss_cache_name

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
                ac1 = ac1[maf_filter]
                ac2 = ac2[maf_filter]

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
            ac1 = ac1[na_mask]
            ac2 = ac2[na_mask]

            if window_size:
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
        summary="""
            Construct a median-joining haplotype network and display it using
            Cytoscape.
        """,
        extended_summary="""
            A haplotype network provides a visualisation of the genetic distance
            between haplotypes. Each node in the network represents a unique
            haplotype. The size (area) of the node is scaled by the number of
            times that unique haplotype was observed within the selected samples.
            A connection between two nodes represents a single SNP difference
            between the corresponding haplotypes.
        """,
    )
    def plot_haplotype_network(
        self,
        region: base_params.regions,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        max_dist: hapnet_params.max_dist = hapnet_params.max_dist_default,
        color: plotly_params.color = None,
        color_discrete_sequence: plotly_params.color_discrete_sequence = None,
        color_discrete_map: plotly_params.color_discrete_map = None,
        category_orders: plotly_params.category_order = None,
        node_size_factor: hapnet_params.node_size_factor = hapnet_params.node_size_factor_default,
        layout: hapnet_params.layout = hapnet_params.layout_default,
        layout_params: Optional[hapnet_params.layout_params] = None,
        server_port: Optional[dash_params.server_port] = None,
        server_mode: Optional[
            dash_params.server_mode
        ] = dash_params.server_mode_default,
        height: dash_params.height = 600,
        width: Optional[dash_params.width] = "100%",
        serve_scripts_locally: dash_params.serve_scripts_locally = dash_params.serve_scripts_locally_default,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ):
        import dash_cytoscape as cyto  # type: ignore
        from dash import Dash, dcc, html  # type: ignore
        from dash.dependencies import Input, Output  # type: ignore

        debug = self._log.debug

        # https://dash.plotly.com/dash-in-jupyter#troubleshooting
        # debug("infer jupyter proxy config")
        # Turn off for now, this seems to crash the kernel!
        # from dash import jupyter_dash
        # jupyter_dash.infer_jupyter_proxy_config()

        if layout != "cose":
            cyto.load_extra_layouts()

        debug("access haplotypes dataset")
        ds_haps = self.haplotypes(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            analysis=analysis,
            chunks=chunks,
            inline_array=inline_array,
        )

        debug("access sample metadata")
        df_samples = self.sample_metadata(
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            sample_sets=sample_sets,
        )

        debug("setup haplotype metadata")
        samples_phased = ds_haps["sample_id"].values
        df_samples_phased = (
            df_samples.set_index("sample_id").loc[samples_phased].reset_index()
        )
        df_haps = df_samples_phased.loc[df_samples_phased.index.repeat(2)].reset_index(
            drop=True
        )

        debug("load haplotypes")
        gt = allel.GenotypeDaskArray(ds_haps["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes"):
            ht = gt.to_haplotypes().compute()

        with self._spinner(desc="Compute haplotype network"):
            debug("count alleles and select segregating sites")
            ac = ht.count_alleles(max_allele=1)
            loc_seg = ac.is_segregating()
            ht_seg = ht[loc_seg]

            debug("identify distinct haplotypes")
            ht_distinct_sets = ht_seg.distinct()
            # find indices of distinct haplotypes - just need one per set
            ht_distinct_indices = [min(s) for s in ht_distinct_sets]
            # reorder by index - TODO is this necessary?
            ix = np.argsort(ht_distinct_indices)
            ht_distinct_indices = [ht_distinct_indices[i] for i in ix]
            ht_distinct_sets = [ht_distinct_sets[i] for i in ix]
            # obtain an array of distinct haplotypes
            ht_distinct = ht_seg.take(ht_distinct_indices, axis=1)
            # count how many observations per distinct haplotype
            ht_counts = [len(s) for s in ht_distinct_sets]

            debug("construct median joining network")
            ht_distinct_mjn, edges, alt_edges = _median_joining_network(
                ht_distinct, max_dist=max_dist
            )
            edges = np.triu(edges)
            alt_edges = np.triu(alt_edges)

            debug("setup colors")
            color_values = None
            color_values_display = None
            color_discrete_map_display = None
            ht_color_counts = None

            if color is not None:
                # Handle string case (direct column name or cohorts_ prefix)
                if isinstance(color, str):
                    # Try direct column name
                    if color in df_haps.columns:
                        color_column = color
                    # Try with cohorts_ prefix
                    elif f"cohorts_{color}" in df_haps.columns:
                        color_column = f"cohorts_{color}"
                    # Neither exists, raise helpful error
                    else:
                        available_columns = ", ".join(df_haps.columns)
                        raise ValueError(
                            f"Column '{color}' or 'cohorts_{color}' not found in sample data. "
                            f"Available columns: {available_columns}"
                        )

                    # Now use the validated color_column for processing
                    df_haps["_partition"] = (
                        df_haps[color_column]
                        .astype(str)
                        .str.replace(r"\W", "", regex=True)
                    )

                    # extract all unique values of the color column
                    color_values = df_haps["_partition"].fillna("<NA>").unique()
                    color_values_mapping = dict(
                        zip(df_haps["_partition"], df_haps[color_column])
                    )
                    color_values_mapping["<NA>"] = "black"
                    color_values_display = [
                        color_values_mapping[c] for c in color_values
                    ]

                # Handle mapping/dictionary case
                elif isinstance(color, Mapping):
                    # For mapping case, we need to create a new column based on the mapping
                    # Initialize with None
                    df_haps["_partition"] = None

                    # Apply each query in the mapping to create the _partition column
                    for label, query in color.items():
                        # Apply the query and assign the label to matching rows
                        mask = df_haps.eval(query)
                        df_haps.loc[mask, "_partition"] = label

                    # Clean up the _partition column to avoid issues with special characters
                    if df_haps["_partition"].notna().any():
                        df_haps["_partition"] = df_haps["_partition"].str.replace(
                            r"\W", "", regex=True
                        )

                    # extract all unique values of the color column
                    color_values = df_haps["_partition"].fillna("<NA>").unique()
                    # For mapping case, use _partition values directly as they're already the labels
                    color_values_mapping = dict(
                        zip(df_haps["_partition"], df_haps["_partition"])
                    )
                    color_values_mapping["<NA>"] = "black"
                    color_values_display = [
                        color_values_mapping[c] for c in color_values
                    ]
                else:
                    # Invalid type
                    raise TypeError(
                        f"Expected color parameter to be a string or mapping, got {type(color).__name__}"
                    )

                # count color values for each distinct haplotype (same for both string and mapping cases)
                ht_color_counts = [
                    df_haps.iloc[list(s)]["_partition"].value_counts().to_dict()
                    for s in ht_distinct_sets
                ]

                # Set up colors (same for both string and mapping cases)
                (
                    color_prepped,
                    color_discrete_map_prepped,
                    category_orders_prepped,
                ) = self._setup_sample_colors_plotly(
                    data=df_haps,
                    color="_partition",
                    color_discrete_map=color_discrete_map,
                    color_discrete_sequence=color_discrete_sequence,
                    category_orders=category_orders,
                )
                del color_discrete_map
                del color_discrete_sequence
                del category_orders
                color_discrete_map_display = {
                    color_values_mapping[v]: c
                    for v, c in color_discrete_map_prepped.items()
                }

        debug("construct graph")
        anon_width = np.sqrt(0.3 * node_size_factor)
        graph_nodes, graph_edges = _mjn_graph(
            ht_distinct=ht_distinct,
            ht_distinct_mjn=ht_distinct_mjn,
            ht_counts=ht_counts,
            ht_color_counts=ht_color_counts,
            color="_partition" if color is not None else None,
            color_values=color_values,
            edges=edges,
            alt_edges=alt_edges,
            node_size_factor=node_size_factor,
            anon_width=anon_width,
        )

        debug("prepare graph data for cytoscape")
        elements = [{"data": n} for n in graph_nodes] + [
            {"data": e} for e in graph_edges
        ]

        debug("define node style")
        node_style = {
            "width": "data(width)",
            "height": "data(width)",
            "pie-size": "100%",
        }
        if color and color_discrete_map_prepped is not None:
            # here are the styles which control the display of nodes as pie
            # charts
            for i, (v, c) in enumerate(color_discrete_map_prepped.items()):
                node_style[f"pie-{i + 1}-background-color"] = c
                node_style[f"pie-{i + 1}-background-size"] = (
                    f"mapData({v}, 0, 100, 0, 100)"
                )
        node_stylesheet = {
            "selector": "node",
            "style": node_style,
        }
        debug(node_stylesheet)

        debug("define edge style")
        edge_stylesheet = {
            "selector": "edge",
            "style": {"curve-style": "bezier", "width": 2, "opacity": 0.5},
        }

        debug("define style for selected node")
        selected_stylesheet = {
            "selector": ":selected",
            "style": {
                "border-width": "3px",
                "border-style": "solid",
                "border-color": "black",
            },
        }

        debug("create figure legend")
        if color is not None:
            legend_fig = _plotly_discrete_legend(
                color="_partition",  # Changed from color=color
                color_values=color_values_display,
                color_discrete_map=color_discrete_map_display,
                category_orders=category_orders_prepped,
            )
            legend_component = dcc.Graph(
                id="legend",
                figure=legend_fig,
                config=dict(
                    displayModeBar=False,
                ),
            )
        else:
            legend_component = html.Div()

        debug("define cytoscape component")
        if layout_params is None:
            graph_layout_params = dict()
        else:
            graph_layout_params = dict(**layout_params)
        graph_layout_params["name"] = layout
        graph_layout_params.setdefault("padding", 10)
        graph_layout_params.setdefault("animate", False)

        cytoscape_component = cyto.Cytoscape(
            id="cytoscape",
            elements=elements,
            layout=graph_layout_params,
            stylesheet=[
                node_stylesheet,
                edge_stylesheet,
                selected_stylesheet,
            ],
            style={
                # width and height needed to get cytoscape component to display
                "width": "100%",
                "height": "100%",
                "background-color": "white",
            },
            # enable selecting multiple nodes with shift click and drag
            boxSelectionEnabled=True,
            # prevent accidentally zooming out to oblivion
            minZoom=0.1,
            # lower scroll wheel zoom sensitivity to prevent accidental zooming when trying to navigate large graphs
            wheelSensitivity=0.1,
        )

        debug("create dash app")
        app = Dash(
            "dash-cytoscape-network",
            # this stylesheet is used to provide support for a rows and columns
            # layout of the components
            external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"],
        )
        # it's generally faster to serve script files from CDN
        app.scripts.config.serve_locally = serve_scripts_locally
        app.layout = html.Div(
            [
                html.Div(
                    cytoscape_component,
                    className="nine columns",
                    style={
                        # required to get cytoscape component to show ...
                        # reduce to prevent scroll overflow
                        "height": f"{height - 50}px",
                        "border": "1px solid black",
                    },
                ),
                html.Div(
                    legend_component,
                    className="three columns",
                    style={
                        "height": f"{height - 50}px",
                    },
                ),
                html.Div(id="output"),
            ],
        )

        debug(
            "define a callback function to display information about the selected node"
        )

        @app.callback(Output("output", "children"), Input("cytoscape", "tapNodeData"))
        def display_tap_node_data(data):
            if data is None:
                return "Click or tap a node for more information."
            else:
                n = data["count"]
                text = f"No. haplotypes: {n}"
                selected_color_data = {
                    color_v_display: int(data.get(color_v, 0) * n / 100)
                    for color_v, color_v_display in zip(
                        color_values, color_values_display
                    )
                }
                selected_color_data = sorted(
                    selected_color_data.items(), key=lambda item: item[1], reverse=True
                )
                color_texts = [
                    f"{color_v}: {color_n}"
                    for color_v, color_n in selected_color_data
                    if color_n > 0
                ]
                if color_texts:
                    color_texts = "; ".join(color_texts)
                    text += f" ({color_texts})"
                return text

        debug("set up run parameters")
        # workaround weird mypy bug here
        run_params: Dict[str, Any] = dict()
        if height is not None:
            run_params["jupyter_height"] = height
        if width is not None:
            run_params["jupyter_width"] = width
        if server_mode is not None:
            run_params["jupyter_mode"] = server_mode
        if server_port is not None:
            run_params["port"] = server_port

        debug("launch the dash app")
        app.run(**run_params)

        debug("launch the dash app")
        app.run(**run_params)
