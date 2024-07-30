import math
import warnings
from abc import abstractmethod
from bisect import bisect_left, bisect_right
from typing import Any, Dict, List, Mapping, Optional, Tuple, Sequence

import allel  # type: ignore
import bokeh.layouts
import bokeh.models
import bokeh.palettes
import bokeh.plotting
import dask
import numba  # type: ignore
import numpy as np
import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objects as go  # type: ignore
import xarray as xr
from numpydoc_decorator import doc  # type: ignore

from malariagen_data.anoph import tree_params
from malariagen_data.anoph.snp_frq import (
    AnophelesSnpFrequencyAnalysis,
    _add_frequency_ci,
    _build_cohorts_from_sample_grouping,
    _prep_samples_for_cohort_grouping,
)

from .anoph import (
    aim_params,
    base_params,
    dash_params,
    diplotype_distance_params,
    gplt_params,
    hapnet_params,
    het_params,
    ihs_params,
    pca_params,
    plotly_params,
    xpehh_params,
)
from .anoph.aim_data import AnophelesAimData
from .anoph.base import AnophelesBase
from .anoph.cnv_data import AnophelesCnvData
from .anoph.genome_features import AnophelesGenomeFeaturesData
from .anoph.genome_sequence import AnophelesGenomeSequenceData
from .anoph.hap_data import AnophelesHapData, hap_params
from .anoph.igv import AnophelesIgv
from .anoph.pca import AnophelesPca
from .anoph.sample_metadata import AnophelesSampleMetadata, locate_cohorts
from .anoph.snp_data import AnophelesSnpData
from .anoph.g123 import AnophelesG123Analysis
from .anoph.fst import AnophelesFstAnalysis
from .anoph.h12 import AnophelesH12Analysis
from .anoph.h1x import AnophelesH1XAnalysis
from .mjn import median_joining_network, mjn_graph
from .anoph.hapclust import AnophelesHapClustAnalysis
from .anoph.dipclust import AnophelesDipClustAnalysis
from .util import (
    CacheMiss,
    Region,
    biallelic_diplotype_cityblock,
    biallelic_diplotype_euclidean,
    biallelic_diplotype_pdist,
    biallelic_diplotype_sqeuclidean,
    check_types,
    jackknife_ci,
    parse_multi_region,
    parse_single_region,
    plotly_discrete_legend,
    region_str,
    simple_xarray_concat,
    pandas_apply,
)

DEFAULT_MAX_COVERAGE_VARIANCE = 0.2


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
    AnophelesSnpFrequencyAnalysis,
    AnophelesPca,
    AnophelesIgv,
    AnophelesAimData,
    AnophelesHapData,
    AnophelesSnpData,
    AnophelesCnvData,
    AnophelesSampleMetadata,
    AnophelesGenomeFeaturesData,
    AnophelesGenomeSequenceData,
    AnophelesBase,
):
    """Anopheles data resources."""

    def __init__(
        self,
        url,
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
        storage_options: Mapping,  # used by fsspec via init_filesystem(url, **kwargs)
        taxon_colors: Optional[Mapping[str, str]],
        virtual_contigs: Optional[Mapping[str, Sequence[str]]],
        gene_names: Optional[Mapping[str, str]],
    ):
        super().__init__(
            url=url,
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
            virtual_contigs=virtual_contigs,
            gene_names=gene_names,
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
        df_roh.rename(columns={"is_marginal": "roh_is_marginal"}, inplace=True)

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

    @check_types
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
    ) -> gplt_params.figure:
        debug = self._log.debug

        # Normalise parameters.
        region_prepped: Region = parse_single_region(self, region)
        del region

        debug("compute windowed heterozygosity")
        sample_id, sample_set, windows, counts = self._sample_count_het(
            sample=sample,
            region=region_prepped,
            site_mask=site_mask,
            window_size=window_size,
            sample_set=sample_set,
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

    @check_types
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
    ) -> gplt_params.figure:
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
        sample_set: Optional[base_params.sample_set] = None,
    ):
        debug = self._log.debug

        debug("access sample metadata, look up sample")
        sample_rec = self.lookup_sample(sample=sample, sample_set=sample_set)
        sample_id = sample_rec.name  # sample_id
        sample_set = sample_rec["sample_set"]

        debug("access SNPs, select data for sample")
        ds_snps = self.snp_calls(
            region=region, sample_sets=sample_set, site_mask=site_mask
        )
        ds_snps_sample = ds_snps.set_index(samples="sample_id").sel(samples=sample_id)

        # snp positions
        pos = ds_snps_sample["variant_position"].values

        # access genotypes
        gt = allel.GenotypeDaskVector(ds_snps_sample["call_genotype"].data)

        # compute het
        with self._dask_progress(desc="Compute heterozygous genotypes"):
            is_het = gt.is_het().compute()

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

    @check_types
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
    ) -> het_params.df_roh:
        debug = self._log.debug

        resolved_region: Region = parse_single_region(self, region)
        del region

        debug("compute windowed heterozygosity")
        sample_id, sample_set, windows, counts = self._sample_count_het(
            sample=sample,
            region=resolved_region,
            site_mask=site_mask,
            window_size=window_size,
            sample_set=sample_set,
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

        return df_roh

    @check_types
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
    ) -> gplt_params.figure:
        debug = self._log.debug

        debug("handle region parameter - this determines the genome region to plot")
        resolved_region: Region = parse_single_region(self, region)
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

    @check_types
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
    ) -> gplt_params.figure:
        debug = self._log.debug

        resolved_region: Region = parse_single_region(self, region)
        del region

        debug("compute windowed heterozygosity")
        sample_id, sample_set, windows, counts = self._sample_count_het(
            sample=sample,
            region=resolved_region,
            site_mask=site_mask,
            window_size=window_size,
            sample_set=sample_set,
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

    def gene_cnv(
        self,
        region: base_params.regions,
        sample_sets=None,
        sample_query=None,
        max_coverage_variance=DEFAULT_MAX_COVERAGE_VARIANCE,
    ):
        """Compute modal copy number by gene, from HMM data.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        sample_sets : str or list of str
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or
            a release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        max_coverage_variance : float, optional
            Remove samples if coverage variance exceeds this value.

        Returns
        -------
        ds : xarray.Dataset
            A dataset of modal copy number per gene and associated data.

        """

        regions: List[Region] = parse_multi_region(self, region)
        del region

        ds = simple_xarray_concat(
            [
                self._gene_cnv(
                    region=r,
                    sample_sets=sample_sets,
                    sample_query=sample_query,
                    max_coverage_variance=max_coverage_variance,
                )
                for r in regions
            ],
            dim="genes",
        )

        return ds

    def _gene_cnv(self, *, region, sample_sets, sample_query, max_coverage_variance):
        debug = self._log.debug

        debug("sanity check")
        assert isinstance(region, Region)

        debug("access HMM data")
        ds_hmm = self.cnv_hmm(
            region=region.contig,
            sample_sets=sample_sets,
            sample_query=sample_query,
            max_coverage_variance=max_coverage_variance,
        )
        pos = ds_hmm["variant_position"].data
        end = ds_hmm["variant_end"].data
        cn = ds_hmm["call_CN"].data
        with self._dask_progress(desc="Load CNV HMM data"):
            pos, end, cn = dask.compute(pos, end, cn)

        debug("access genes")
        df_genome_features = self.genome_features(region=region)
        df_genes = df_genome_features.query(f"type == '{self._gff_gene_type}'")

        debug("setup intermediates")
        windows = []
        modes = []
        counts = []

        debug("iterate over genes")
        genes_iterator = self._progress(
            df_genes.itertuples(),
            desc="Compute modal gene copy number",
            total=len(df_genes),
        )
        for gene in genes_iterator:
            # locate windows overlapping the gene
            loc_gene_start = bisect_left(end, gene.start)
            loc_gene_stop = bisect_right(pos, gene.end)
            w = loc_gene_stop - loc_gene_start
            windows.append(w)

            # slice out copy number data for the given gene
            cn_gene = cn[loc_gene_start:loc_gene_stop]

            # compute the modes
            m, c = _cn_mode(cn_gene, vmax=12)
            modes.append(m)
            counts.append(c)

        debug("combine results")
        windows = np.array(windows)
        modes = np.vstack(modes)
        counts = np.vstack(counts)

        debug("build dataset")
        ds_out = xr.Dataset(
            coords={
                "gene_id": (["genes"], df_genes["ID"].values),
                "sample_id": (["samples"], ds_hmm["sample_id"].values),
            },
            data_vars={
                "gene_contig": (["genes"], df_genes["contig"].values),
                "gene_start": (["genes"], df_genes["start"].values),
                "gene_end": (["genes"], df_genes["end"].values),
                "gene_windows": (["genes"], windows),
                "gene_name": (
                    ["genes"],
                    df_genes[self._gff_gene_name_attribute].values,
                ),
                "gene_strand": (["genes"], df_genes["strand"].values),
                "gene_description": (["genes"], df_genes["description"].values),
                "CN_mode": (["genes", "samples"], modes),
                "CN_mode_count": (["genes", "samples"], counts),
                "sample_coverage_variance": (
                    ["samples"],
                    ds_hmm["sample_coverage_variance"].values,
                ),
                "sample_is_high_variance": (
                    ["samples"],
                    ds_hmm["sample_is_high_variance"].values,
                ),
            },
        )

        return ds_out

    def gene_cnv_frequencies(
        self,
        region: base_params.regions,
        cohorts,
        sample_query=None,
        min_cohort_size=10,
        sample_sets=None,
        drop_invariant=True,
        max_coverage_variance=DEFAULT_MAX_COVERAGE_VARIANCE,
    ):
        """Compute modal copy number by gene, then compute the frequency of
        amplifications and deletions in one or more cohorts, from HMM data.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        cohorts : str or dict
            If a string, gives the name of a predefined cohort set, e.g., one of
            {"admin1_month", "admin1_year", "admin2_month", "admin2_year"}.
            If a dict, should map cohort labels to sample queries, e.g.,
            ``{"bf_2012_col": "country == 'Burkina Faso' and year == 2012 and
            taxon == 'coluzzii'"}``.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int
            Minimum cohort size, below which cohorts are dropped.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        drop_invariant : bool, optional
            If True, drop any rows where there is no evidence of variation.
        max_coverage_variance : float, optional
            Remove samples if coverage variance exceeds this value.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of CNV amplification (amp) and deletion (del)
            frequencies in the specified cohorts, one row per gene and CNV type
            (amp/del).

        """
        debug = self._log.debug

        debug("check and normalise parameters")
        regions: List[Region] = parse_multi_region(self, region)
        del region

        debug("access and concatenate data from regions")
        df = pd.concat(
            [
                self._gene_cnv_frequencies(
                    region=r,
                    cohorts=cohorts,
                    sample_query=sample_query,
                    min_cohort_size=min_cohort_size,
                    sample_sets=sample_sets,
                    drop_invariant=drop_invariant,
                    max_coverage_variance=max_coverage_variance,
                )
                for r in regions
            ],
            axis=0,
        )

        debug("add metadata")
        title = f"Gene CNV frequencies ({region_str(regions)})"
        df.attrs["title"] = title

        return df

    def _gene_cnv_frequencies(
        self,
        *,
        region,
        cohorts,
        sample_query,
        min_cohort_size,
        sample_sets,
        drop_invariant,
        max_coverage_variance,
    ):
        debug = self._log.debug

        debug("sanity check - this function is one region at a time")
        assert isinstance(region, Region)

        debug("get gene copy number data")
        ds_cnv = self.gene_cnv(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            max_coverage_variance=max_coverage_variance,
        )

        debug("load sample metadata")
        df_samples = self.sample_metadata(sample_sets=sample_sets)

        debug("align sample metadata with samples in CNV data")
        sample_id = ds_cnv["sample_id"].values
        df_samples = df_samples.set_index("sample_id").loc[sample_id].reset_index()

        debug("figure out expected copy number")
        if region.contig == "X":
            is_male = (df_samples["sex_call"] == "M").values
            expected_cn = np.where(is_male, 1, 2)[np.newaxis, :]
        else:
            expected_cn = 2

        debug(
            "setup output dataframe - two rows for each gene, one for amplification and one for deletion"
        )
        n_genes = ds_cnv.sizes["genes"]
        df_genes = ds_cnv[
            [
                "gene_id",
                "gene_name",
                "gene_strand",
                "gene_description",
                "gene_contig",
                "gene_start",
                "gene_end",
            ]
        ].to_dataframe()
        df = pd.concat([df_genes, df_genes], axis=0).reset_index(drop=True)
        df.rename(
            columns={
                "gene_contig": "contig",
                "gene_start": "start",
                "gene_end": "end",
            },
            inplace=True,
        )

        debug("add CNV type column")
        df_cnv_type = pd.DataFrame(
            {
                "cnv_type": np.array(
                    (["amp"] * n_genes) + (["del"] * n_genes), dtype=object
                )
            }
        )
        df = pd.concat([df, df_cnv_type], axis=1)

        debug("set up intermediates")
        cn = ds_cnv["CN_mode"].values
        is_amp = cn > expected_cn
        is_del = (cn >= 0) & (cn < expected_cn)
        is_called = cn >= 0

        debug("set up cohort dict")
        coh_dict = locate_cohorts(cohorts=cohorts, data=df_samples)

        debug("compute cohort frequencies")
        freq_cols = dict()
        for coh, loc_coh in coh_dict.items():
            n_samples = np.count_nonzero(loc_coh)
            debug(f"{coh}, {n_samples} samples")

            if n_samples >= min_cohort_size:
                # subset data to cohort
                is_amp_coh = np.compress(loc_coh, is_amp, axis=1)
                is_del_coh = np.compress(loc_coh, is_del, axis=1)
                is_called_coh = np.compress(loc_coh, is_called, axis=1)

                # count amplifications and deletions
                amp_count_coh = np.sum(is_amp_coh, axis=1)
                del_count_coh = np.sum(is_del_coh, axis=1)
                called_count_coh = np.sum(is_called_coh, axis=1)

                # compute frequencies, taking accessibility into account
                with np.errstate(divide="ignore", invalid="ignore"):
                    amp_freq_coh = np.where(
                        called_count_coh > 0, amp_count_coh / called_count_coh, np.nan
                    )
                    del_freq_coh = np.where(
                        called_count_coh > 0, del_count_coh / called_count_coh, np.nan
                    )

                freq_cols[f"frq_{coh}"] = np.concatenate([amp_freq_coh, del_freq_coh])

        debug("build a dataframe with the frequency columns")
        df_freqs = pd.DataFrame(freq_cols)

        debug("compute max_af and additional columns")
        df_extras = pd.DataFrame(
            {
                "max_af": df_freqs.max(axis=1),
                "windows": np.concatenate(
                    [ds_cnv["gene_windows"].values, ds_cnv["gene_windows"].values]
                ),
            }
        )

        debug("build the final dataframe")
        df.reset_index(drop=True, inplace=True)
        df = pd.concat([df, df_freqs, df_extras], axis=1)
        df.sort_values(["contig", "start", "cnv_type"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        debug("add label")
        df["label"] = pandas_apply(
            self._make_gene_cnv_label, df, columns=["gene_id", "gene_name", "cnv_type"]
        )

        debug("deal with invariants")
        if drop_invariant:
            df = df.query("max_af > 0")

        debug("set index for convenience")
        df.set_index(["gene_id", "gene_name", "cnv_type"], inplace=True)

        return df

    def gene_cnv_frequencies_advanced(
        self,
        region: base_params.regions,
        area_by,
        period_by,
        sample_sets=None,
        sample_query=None,
        min_cohort_size=10,
        variant_query=None,
        drop_invariant=True,
        max_coverage_variance=DEFAULT_MAX_COVERAGE_VARIANCE,
        ci_method="wilson",
    ):
        """Group samples by taxon, area (space) and period (time), then compute
        gene CNV counts and frequencies.

        Parameters
        ----------
        region: str or list of str or Region or list of Region
            Chromosome arm (e.g., "2L"), gene name (e.g., "AGAP007280"), genomic
            region defined with coordinates (e.g., "2L:44989425-44998059") or a
            named tuple with genomic location `Region(contig, start, end)`.
            Multiple values can be provided as a list, in which case data will
            be concatenated, e.g., ["3R", "3L"].
        area_by : str
            Column name in the sample metadata to use to group samples spatially.
            E.g., use "admin1_iso" or "admin1_name" to group by level 1
            administrative divisions, or use "admin2_name" to group by level 2
            administrative divisions.
        period_by : {"year", "quarter", "month"}
            Length of time to group samples temporally.
        sample_sets : str or list of str, optional
            Can be a sample set identifier (e.g., "AG1000G-AO") or a list of
            sample set identifiers (e.g., ["AG1000G-BF-A", "AG1000G-BF-B"]) or a
            release identifier (e.g., "3.0") or a list of release identifiers.
        sample_query : str, optional
            A pandas query string which will be evaluated against the sample
            metadata e.g., "taxon == 'coluzzii' and country == 'Burkina Faso'".
        min_cohort_size : int, optional
            Minimum cohort size. Any cohorts below this size are omitted.
        variant_query : str, optional
            A pandas query string which will be evaluated against variants.
        drop_invariant : bool, optional
            If True, drop any rows where there is no evidence of variation.
        max_coverage_variance : float, optional
            Remove samples if coverage variance exceeds this value.
        ci_method : {"normal", "agresti_coull", "beta", "wilson", "binom_test"}, optional
            Method to use for computing confidence intervals, passed through to
            `statsmodels.stats.proportion.proportion_confint`.

        Returns
        -------
        ds : xarray.Dataset
            The resulting dataset contains data has dimensions "cohorts" and
            "variants". Variables prefixed with "cohort" are 1-dimensional
            arrays with data about the cohorts, such as the area, period, taxon
            and cohort size. Variables prefixed with "variant" are 1-dimensional
            arrays with data about the variants, such as the contig, position,
            reference and alternate alleles. Variables prefixed with "event" are
            2-dimensional arrays with the allele counts and frequency
            calculations.

        """

        regions: List[Region] = parse_multi_region(self, region)
        del region

        ds = simple_xarray_concat(
            [
                self._gene_cnv_frequencies_advanced(
                    region=r,
                    area_by=area_by,
                    period_by=period_by,
                    sample_sets=sample_sets,
                    sample_query=sample_query,
                    min_cohort_size=min_cohort_size,
                    variant_query=variant_query,
                    drop_invariant=drop_invariant,
                    max_coverage_variance=max_coverage_variance,
                    ci_method=ci_method,
                )
                for r in regions
            ],
            dim="variants",
        )

        title = f"Gene CNV frequencies ({region_str(regions)})"
        ds.attrs["title"] = title

        return ds

    def _gene_cnv_frequencies_advanced(
        self,
        *,
        region,
        area_by,
        period_by,
        sample_sets,
        sample_query,
        min_cohort_size,
        variant_query,
        drop_invariant,
        max_coverage_variance,
        ci_method,
    ):
        debug = self._log.debug

        debug("sanity check - here we deal with one region only")
        assert isinstance(region, Region)

        debug("access gene CNV calls")
        ds_cnv = self.gene_cnv(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            max_coverage_variance=max_coverage_variance,
        )

        debug("load sample metadata")
        df_samples = self.sample_metadata(sample_sets=sample_sets)

        debug("align sample metadata")
        sample_id = ds_cnv["sample_id"].values
        df_samples = df_samples.set_index("sample_id").loc[sample_id].reset_index()

        debug("prepare sample metadata for cohort grouping")
        df_samples = _prep_samples_for_cohort_grouping(
            df_samples=df_samples,
            area_by=area_by,
            period_by=period_by,
        )

        debug("group samples to make cohorts")
        group_samples_by_cohort = df_samples.groupby(["taxon", "area", "period"])

        debug("build cohorts dataframe")
        df_cohorts = _build_cohorts_from_sample_grouping(
            group_samples_by_cohort=group_samples_by_cohort,
            min_cohort_size=min_cohort_size,
        )

        debug("figure out expected copy number")
        if region.contig == "X":
            is_male = (df_samples["sex_call"] == "M").values
            expected_cn = np.where(is_male, 1, 2)[np.newaxis, :]
        else:
            expected_cn = 2

        debug("set up intermediates")
        cn = ds_cnv["CN_mode"].values
        is_amp = cn > expected_cn
        is_del = (cn >= 0) & (cn < expected_cn)
        is_called = cn >= 0

        debug("set up main event variables")
        n_genes = ds_cnv.sizes["genes"]
        n_variants, n_cohorts = n_genes * 2, len(df_cohorts)
        count = np.zeros((n_variants, n_cohorts), dtype=int)
        nobs = np.zeros((n_variants, n_cohorts), dtype=int)

        debug("build event count and nobs for each cohort")
        for cohort_index, cohort in enumerate(df_cohorts.itertuples()):
            # construct grouping key
            cohort_key = cohort.taxon, cohort.area, cohort.period

            # obtain sample indices for cohort
            sample_indices = group_samples_by_cohort.indices[cohort_key]

            # select genotype data for cohort
            cohort_is_amp = np.take(is_amp, sample_indices, axis=1)
            cohort_is_del = np.take(is_del, sample_indices, axis=1)
            cohort_is_called = np.take(is_called, sample_indices, axis=1)

            # compute cohort allele counts
            np.sum(cohort_is_amp, axis=1, out=count[::2, cohort_index])
            np.sum(cohort_is_del, axis=1, out=count[1::2, cohort_index])

            # compute cohort allele numbers
            cohort_n_called = np.sum(cohort_is_called, axis=1)
            nobs[:, cohort_index] = np.repeat(cohort_n_called, 2)

        debug("compute frequency")
        with np.errstate(divide="ignore", invalid="ignore"):
            # ignore division warnings
            frequency = np.where(nobs > 0, count / nobs, np.nan)

        debug("make dataframe of variants")
        with warnings.catch_warnings():
            # ignore "All-NaN slice encountered" warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_af = np.nanmax(frequency, axis=1)
        df_variants = pd.DataFrame(
            {
                "contig": region.contig,
                "start": np.repeat(ds_cnv["gene_start"].values, 2),
                "end": np.repeat(ds_cnv["gene_end"].values, 2),
                "windows": np.repeat(ds_cnv["gene_windows"].values, 2),
                # alternate amplification and deletion
                "cnv_type": np.tile(np.array(["amp", "del"]), n_genes),
                "max_af": max_af,
                "gene_id": np.repeat(ds_cnv["gene_id"].values, 2),
                "gene_name": np.repeat(ds_cnv["gene_name"].values, 2),
                "gene_strand": np.repeat(ds_cnv["gene_strand"].values, 2),
            }
        )

        debug("add variant label")
        df_variants["label"] = pandas_apply(
            self._make_gene_cnv_label,
            df_variants,
            columns=["gene_id", "gene_name", "cnv_type"],
        )

        debug("build the output dataset")
        ds_out = xr.Dataset()

        debug("cohort variables")
        for coh_col in df_cohorts.columns:
            ds_out[f"cohort_{coh_col}"] = "cohorts", df_cohorts[coh_col]

        debug("variant variables")
        for snp_col in df_variants.columns:
            ds_out[f"variant_{snp_col}"] = "variants", df_variants[snp_col]

        debug("event variables")
        ds_out["event_count"] = ("variants", "cohorts"), count
        ds_out["event_nobs"] = ("variants", "cohorts"), nobs
        ds_out["event_frequency"] = ("variants", "cohorts"), frequency

        debug("deal with invariants")
        if drop_invariant:
            loc_variant = df_variants["max_af"].values > 0
            ds_out = ds_out.isel(variants=loc_variant)
            df_variants = df_variants.loc[loc_variant].reset_index(drop=True)

        debug("apply variant query")
        if variant_query is not None:
            loc_variants = df_variants.eval(variant_query).values
            ds_out = ds_out.isel(variants=loc_variants)

        debug("add confidence intervals")
        _add_frequency_ci(ds=ds_out, ci_method=ci_method)

        debug("tidy up display by sorting variables")
        ds_out = ds_out[sorted(ds_out)]

        return ds_out

    def _block_jackknife_cohort_diversity_stats(
        self, *, cohort_label, ac, n_jack, confidence_level
    ):
        debug = self._log.debug

        debug("set up for diversity calculations")
        n_sites = ac.shape[0]
        ac = allel.AlleleCountsArray(ac)
        n = ac.sum(axis=1).max()  # number of chromosomes sampled
        n_sites = min(n_sites, ac.shape[0])  # number of sites
        block_length = n_sites // n_jack  # number of sites in each block
        n_sites_j = n_sites - block_length  # number of sites in each jackknife resample

        debug("compute scaling constants")
        a1 = np.sum(1 / np.arange(1, n))
        a2 = np.sum(1 / (np.arange(1, n) ** 2))
        b1 = (n + 1) / (3 * (n - 1))
        b2 = 2 * (n**2 + n + 3) / (9 * n * (n - 1))
        c1 = b1 - (1 / a1)
        c2 = b2 - ((n + 2) / (a1 * n)) + (a2 / (a1**2))
        e1 = c1 / a1
        e2 = c2 / (a1**2 + a2)

        debug(
            "compute some intermediates ahead of time, to minimise computation during jackknife resampling"
        )
        mpd_data = allel.mean_pairwise_difference(ac, fill=0)
        # N.B., here we compute the number of segregating sites as the number
        # of alleles minus 1. This follows the sgkit and tskit implementations,
        # and is different from scikit-allel.
        seg_data = ac.allelism() - 1

        debug("compute estimates from all data")
        theta_pi_abs_data = np.sum(mpd_data)
        theta_pi_data = theta_pi_abs_data / n_sites
        S_data = np.sum(seg_data)
        theta_w_abs_data = S_data / a1
        theta_w_data = theta_w_abs_data / n_sites
        d_data = theta_pi_abs_data - theta_w_abs_data
        d_stdev_data = np.sqrt((e1 * S_data) + (e2 * S_data * (S_data - 1)))
        tajima_d_data = d_data / d_stdev_data

        debug("set up for jackknife resampling")
        jack_theta_pi = []
        jack_theta_w = []
        jack_tajima_d = []

        debug("begin jackknife resampling")
        for i in range(n_jack):
            # locate block to delete
            block_start = i * block_length
            block_stop = block_start + block_length
            loc_j = np.ones(n_sites, dtype=bool)
            loc_j[block_start:block_stop] = False
            assert np.count_nonzero(loc_j) == n_sites_j

            # resample data and compute statistics

            # theta_pi
            mpd_j = mpd_data[loc_j]
            theta_pi_abs_j = np.sum(mpd_j)
            theta_pi_j = theta_pi_abs_j / n_sites_j
            jack_theta_pi.append(theta_pi_j)

            # theta_w
            seg_j = seg_data[loc_j]
            S_j = np.sum(seg_j)
            theta_w_abs_j = S_j / a1
            theta_w_j = theta_w_abs_j / n_sites_j
            jack_theta_w.append(theta_w_j)

            # tajima_d
            d_j = theta_pi_abs_j - theta_w_abs_j
            d_stdev_j = np.sqrt((e1 * S_j) + (e2 * S_j * (S_j - 1)))
            tajima_d_j = d_j / d_stdev_j
            jack_tajima_d.append(tajima_d_j)

        # calculate jackknife stats
        (
            theta_pi_estimate,
            theta_pi_bias,
            theta_pi_std_err,
            theta_pi_ci_err,
            theta_pi_ci_low,
            theta_pi_ci_upp,
        ) = jackknife_ci(
            stat_data=theta_pi_data,
            jack_stat=jack_theta_pi,
            confidence_level=confidence_level,
        )
        (
            theta_w_estimate,
            theta_w_bias,
            theta_w_std_err,
            theta_w_ci_err,
            theta_w_ci_low,
            theta_w_ci_upp,
        ) = jackknife_ci(
            stat_data=theta_w_data,
            jack_stat=jack_theta_w,
            confidence_level=confidence_level,
        )
        (
            tajima_d_estimate,
            tajima_d_bias,
            tajima_d_std_err,
            tajima_d_ci_err,
            tajima_d_ci_low,
            tajima_d_ci_upp,
        ) = jackknife_ci(
            stat_data=tajima_d_data,
            jack_stat=jack_tajima_d,
            confidence_level=confidence_level,
        )

        return dict(
            cohort=cohort_label,
            theta_pi=theta_pi_data,
            theta_pi_estimate=theta_pi_estimate,
            theta_pi_bias=theta_pi_bias,
            theta_pi_std_err=theta_pi_std_err,
            theta_pi_ci_err=theta_pi_ci_err,
            theta_pi_ci_low=theta_pi_ci_low,
            theta_pi_ci_upp=theta_pi_ci_upp,
            theta_w=theta_w_data,
            theta_w_estimate=theta_w_estimate,
            theta_w_bias=theta_w_bias,
            theta_w_std_err=theta_w_std_err,
            theta_w_ci_err=theta_w_ci_err,
            theta_w_ci_low=theta_w_ci_low,
            theta_w_ci_upp=theta_w_ci_upp,
            tajima_d=tajima_d_data,
            tajima_d_estimate=tajima_d_estimate,
            tajima_d_bias=tajima_d_bias,
            tajima_d_std_err=tajima_d_std_err,
            tajima_d_ci_err=tajima_d_ci_err,
            tajima_d_ci_low=tajima_d_ci_low,
            tajima_d_ci_upp=tajima_d_ci_upp,
        )

    @check_types
    @doc(
        summary="""
            Compute genetic diversity summary statistics for a cohort of
            individuals.
        """,
        returns="""
            A pandas series with summary statistics and their confidence
            intervals.
        """,
    )
    def cohort_diversity_stats(
        self,
        cohort: base_params.cohort,
        cohort_size: base_params.cohort_size,
        region: base_params.regions,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        site_class: Optional[base_params.site_class] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        random_seed: base_params.random_seed = 42,
        n_jack: base_params.n_jack = 200,
        confidence_level: base_params.confidence_level = 0.95,
    ) -> pd.Series:
        debug = self._log.debug

        debug("process cohort parameter")
        cohort_query = None
        if isinstance(cohort, str):
            # assume it is one of the predefined cohorts
            cohort_label = cohort
            df_samples = self.sample_metadata(sample_sets=sample_sets)
            cohort_cols = [c for c in df_samples.columns if c.startswith("cohort_")]
            for c in cohort_cols:
                if cohort in set(df_samples[c]):
                    cohort_query = f"{c} == '{cohort}'"
                    break
            if cohort_query is None:
                raise ValueError(f"unknown cohort: {cohort}")

        elif isinstance(cohort, (list, tuple)) and len(cohort) == 2:
            cohort_label, cohort_query = cohort

        else:
            raise TypeError(r"invalid cohort parameter: {cohort!r}")

        debug("access allele counts")
        ac = self.snp_allele_counts(
            region=region,
            site_mask=site_mask,
            site_class=site_class,
            sample_query=cohort_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        debug("compute diversity stats")
        stats = self._block_jackknife_cohort_diversity_stats(
            cohort_label=cohort_label,
            ac=ac,
            n_jack=n_jack,
            confidence_level=confidence_level,
        )

        debug("compute some extra cohort variables")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=cohort_query
        )
        extra_fields = [
            ("taxon", "unique"),
            ("year", "unique"),
            ("month", "unique"),
            ("country", "unique"),
            ("admin1_iso", "unique"),
            ("admin1_name", "unique"),
            ("admin2_name", "unique"),
            ("longitude", "mean"),
            ("latitude", "mean"),
        ]
        for field, agg in extra_fields:
            if agg == "unique":
                vals = df_samples[field].sort_values().unique()
                if len(vals) == 0:
                    val = np.nan
                elif len(vals) == 1:
                    val = vals[0]
                else:
                    val = vals.tolist()
            elif agg == "mean":
                vals = df_samples[field]
                if len(vals) == 0:
                    val = np.nan
                else:
                    val = np.mean(vals)
            else:
                val = np.nan
            stats[field] = val

        return pd.Series(stats)

    @check_types
    @doc(
        summary="""
            Compute genetic diversity summary statistics for multiple cohorts.
        """,
        returns="""
            A DataFrame where each row provides summary statistics and their
            confidence intervals for a single cohort.
        """,
    )
    def diversity_stats(
        self,
        cohorts: base_params.cohorts,
        cohort_size: base_params.cohort_size,
        region: base_params.regions,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        site_class: Optional[base_params.site_class] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        random_seed: base_params.random_seed = 42,
        n_jack: base_params.n_jack = 200,
        confidence_level: base_params.confidence_level = 0.95,
    ) -> pd.DataFrame:
        # Normalise cohorts parameter.
        cohort_queries = self._setup_cohort_queries(
            cohorts=cohorts,
            sample_sets=sample_sets,
            sample_query=sample_query,
            cohort_size=cohort_size,
            min_cohort_size=None,
        )

        # Compute diversity stats for cohorts.
        all_stats = []
        for cohort_label, cohort_query in cohort_queries.items():
            stats = self.cohort_diversity_stats(
                cohort=(cohort_label, cohort_query),
                cohort_size=cohort_size,
                region=region,
                site_mask=site_mask,
                site_class=site_class,
                sample_sets=sample_sets,
                random_seed=random_seed,
                n_jack=n_jack,
                confidence_level=confidence_level,
            )
            all_stats.append(stats)
        df_stats = pd.DataFrame(all_stats)

        return df_stats

    @check_types
    @doc(
        summary="Plot diversity summary statistics for multiple cohorts.",
        parameters=dict(
            df_stats="Output from `diversity_stats()`.",
            bar_plot_height="Height of bar plots in pixels (px).",
            bar_width="Width per bar in pixels (px).",
            scatter_plot_height="Height of scatter plot in pixels (px).",
            scatter_plot_width="Width of scatter plot in pixels (px).",
            plot_kwargs="Extra plotting parameters.",
        ),
    )
    def plot_diversity_stats(
        self,
        df_stats: pd.DataFrame,
        color: plotly_params.color = None,
        bar_plot_height: int = 450,
        bar_width: int = 30,
        scatter_plot_height: int = 500,
        scatter_plot_width: int = 500,
        template: plotly_params.template = "plotly_white",
        color_discrete_sequence: plotly_params.color_discrete_sequence = None,
        color_discrete_map: plotly_params.color_discrete_map = None,
        category_orders: plotly_params.category_order = None,
        plot_kwargs: Optional[Mapping] = None,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
    ) -> Optional[Tuple[go.Figure, ...]]:
        # Handle color.
        (
            color_prepped,
            color_discrete_map_prepped,
            category_orders_prepped,
        ) = self._setup_sample_colors_plotly(
            data=df_stats,
            color=color,
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=color_discrete_sequence,
            category_orders=category_orders,
        )
        del color
        del color_discrete_map
        del color_discrete_sequence
        del category_orders

        # Set up common plotting parameters.
        default_plot_kwargs = dict(
            hover_name="cohort",
            hover_data=[
                "taxon",
                "country",
                "admin1_iso",
                "admin1_name",
                "admin2_name",
                "longitude",
                "latitude",
                "year",
                "month",
            ],
            labels={
                "theta_pi_estimate": "θ<sub>π</sub>",
                "theta_w_estimate": "θ<sub>𝑤</sub>",
                "tajima_d_estimate": "𝐷",
                "cohort": "Cohort",
                "taxon": "Taxon",
                "country": "Country",
            },
            color=color_prepped,
            color_discrete_map=color_discrete_map_prepped,
            category_orders=category_orders_prepped,
            template=template,
        )

        # Finalise parameters.
        if plot_kwargs is None:
            plot_kwargs = dict()
        default_plot_kwargs.update(plot_kwargs)
        plot_kwargs = default_plot_kwargs
        bar_plot_width = 300 + bar_width * len(df_stats)

        # Nucleotide diversity bar plot.
        fig1 = px.bar(
            data_frame=df_stats,
            x="cohort",
            y="theta_pi_estimate",
            error_y="theta_pi_ci_err",
            title="Nucleotide diversity",
            height=bar_plot_height,
            width=bar_plot_width,
            **plot_kwargs,
        )

        # Watterson's estimator bar plot.
        fig2 = px.bar(
            data_frame=df_stats,
            x="cohort",
            y="theta_w_estimate",
            error_y="theta_w_ci_err",
            title="Watterson's estimator",
            height=bar_plot_height,
            width=bar_plot_width,
            **plot_kwargs,
        )

        # Tajima's D bar plot.
        fig3 = px.bar(
            data_frame=df_stats,
            x="cohort",
            y="tajima_d_estimate",
            error_y="tajima_d_ci_err",
            title="Tajima's D",
            height=bar_plot_height,
            width=bar_plot_width,
            **plot_kwargs,
        )

        # Scatter plot comparing diversity estimators.
        fig4 = px.scatter(
            data_frame=df_stats,
            x="theta_pi_estimate",
            y="theta_w_estimate",
            error_x="theta_pi_ci_err",
            error_y="theta_w_ci_err",
            title="Diversity estimators",
            width=scatter_plot_width,
            height=scatter_plot_height,
            **plot_kwargs,
        )

        if show:  # pragma: no cover
            fig1.show(renderer=renderer)
            fig2.show(renderer=renderer)
            fig3.show(renderer=renderer)
            fig4.show(renderer=renderer)
            return None
        else:
            return (fig1, fig2, fig3, fig4)

    @check_types
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
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._ihs_gwss(**params)
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
    ):
        ds_haps = self.haplotypes(
            region=contig,
            analysis=analysis,
            sample_query=sample_query,
            sample_sets=sample_sets,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
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

    @check_types
    @doc(
        summary="Run and plot iHS GWSS data.",
    )
    def plot_ihs_gwss_track(
        self,
        contig: base_params.contig,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
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
    ) -> gplt_params.figure:
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
            sample_sets=sample_sets,
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

    @check_types
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
    ) -> gplt_params.figure:
        # gwss track
        fig1 = self.plot_xpehh_gwss_track(
            contig=contig,
            analysis=analysis,
            sample_sets=sample_sets,
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
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

    @doc(
        summary="Run and plot iHS GWSS data.",
    )
    def plot_ihs_gwss(
        self,
        contig: base_params.contig,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
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
    ) -> gplt_params.figure:
        # gwss track
        fig1 = self.plot_ihs_gwss_track(
            contig=contig,
            analysis=analysis,
            sample_sets=sample_sets,
            sample_query=sample_query,
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
            cohort1_query=cohort1_query,
            cohort2_query=cohort2_query,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._xpehh_gwss(**params)  # self.
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
    ):
        ds_haps1 = self.haplotypes(
            region=contig,
            analysis=analysis,
            sample_query=cohort1_query,
            sample_sets=sample_sets,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
        )

        ds_haps2 = self.haplotypes(
            region=contig,
            analysis=analysis,
            sample_query=cohort2_query,
            sample_sets=sample_sets,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
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
    ) -> gplt_params.figure:
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
            sample_sets=sample_sets,
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

    @check_types
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
            analysis=analysis,
        )

        debug("access sample metadata")
        df_samples = self.sample_metadata(
            sample_query=sample_query, sample_sets=sample_sets
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
            ht_distinct_mjn, edges, alt_edges = median_joining_network(
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
            # sanitise color column - necessary to avoid grey pie chart segments
            df_haps["partition"] = df_haps[color].str.replace(r"\W", "", regex=True)

            # extract all unique values of the color column
            color_values = df_haps["partition"].unique()
            color_values_mapping = dict(zip(df_haps["partition"], df_haps[color]))
            color_values_display = [color_values_mapping[c] for c in color_values]

            # count color values for each distinct haplotype
            ht_color_counts = [
                df_haps.iloc[list(s)]["partition"].value_counts().to_dict()
                for s in ht_distinct_sets
            ]

            # Set up colors.
            (
                color_prepped,
                color_discrete_map_prepped,
                category_orders_prepped,
            ) = self._setup_sample_colors_plotly(
                data=df_haps,
                color="partition",
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
        graph_nodes, graph_edges = mjn_graph(
            ht_distinct=ht_distinct,
            ht_distinct_mjn=ht_distinct_mjn,
            ht_counts=ht_counts,
            ht_color_counts=ht_color_counts,
            color=color,
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
                node_style[
                    f"pie-{i + 1}-background-size"
                ] = f"mapData({v}, 0, 100, 0, 100)"
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
            legend_fig = plotly_discrete_legend(
                color=color,
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

    @doc(
        summary="""
            Compute pairwise distances between samples using biallelic SNP genotypes.
        """,
        parameters=dict(
            metric="Distance metric, one of 'cityblock', 'euclidean' or 'sqeuclidean'.",
        ),
    )
    def biallelic_diplotype_pairwise_distances(
        self,
        region: base_params.regions,
        n_snps: base_params.n_snps,
        metric: diplotype_distance_params.distance_metric = "cityblock",
        thin_offset: base_params.thin_offset = 0,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_indices: Optional[base_params.sample_indices] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        site_class: Optional[base_params.site_class] = None,
        min_minor_ac: Optional[base_params.min_minor_ac] = None,
        max_missing_an: Optional[base_params.max_missing_an] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
        random_seed: base_params.random_seed = 42,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "biallelic_diplotype_pairwise_distances"

        # Normalize params for consistent hash value.
        (
            sample_sets_prepped,
            sample_indices_prepped,
        ) = self._prep_sample_selection_cache_params(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_indices=sample_indices,
        )
        region_prepped = self._prep_region_cache_param(region=region)
        site_mask_prepped = self._prep_optional_site_mask_param(site_mask=site_mask)
        del sample_sets
        del sample_query
        del sample_indices
        del region
        del site_mask
        params = dict(
            region=region_prepped,
            n_snps=n_snps,
            metric=metric,
            thin_offset=thin_offset,
            sample_sets=sample_sets_prepped,
            sample_indices=sample_indices_prepped,
            site_mask=site_mask_prepped,
            site_class=site_class,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
        )

        # Try to retrieve results from the cache.
        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._biallelic_diplotype_pairwise_distances(
                inline_array=inline_array, chunks=chunks, **params
            )
            self.results_cache_set(name=name, params=params, results=results)

        # Unpack results.
        dist: np.ndarray = results["dist"]
        samples: np.ndarray = results["samples"]
        n_snps_used: int = int(results["n_snps"][()])  # ensure scalar

        return dist, samples, n_snps_used

    def _biallelic_diplotype_pairwise_distances(
        self,
        region,
        n_snps,
        metric,
        thin_offset,
        sample_sets,
        sample_indices,
        site_mask,
        site_class,
        inline_array,
        chunks,
        cohort_size,
        min_cohort_size,
        max_cohort_size,
        random_seed,
        min_minor_ac,
        max_missing_an,
    ):
        # Compute diplotypes.
        gn, samples = self.biallelic_diplotypes(
            region=region,
            sample_sets=sample_sets,
            sample_indices=sample_indices,
            site_mask=site_mask,
            site_class=site_class,
            inline_array=inline_array,
            chunks=chunks,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            max_missing_an=max_missing_an,
            min_minor_ac=min_minor_ac,
            n_snps=n_snps,
            thin_offset=thin_offset,
        )

        # Record number of SNPs used.
        n_snps = gn.shape[0]

        # Prepare data for pairwise distance calculation.
        X = np.ascontiguousarray(gn.T)

        # Look up distance function.
        if metric == "cityblock":
            distfun = biallelic_diplotype_cityblock
        elif metric == "sqeuclidean":
            distfun = biallelic_diplotype_sqeuclidean
        elif metric == "euclidean":
            distfun = biallelic_diplotype_euclidean
        else:
            raise ValueError("Unsupported metric.")

        with self._spinner("Compute pairwise distances"):
            dist = biallelic_diplotype_pdist(X, distfun=distfun)

        return dict(
            dist=dist,
            samples=samples,
            n_snps=np.array(
                n_snps
            ),  # ensure consistent behaviour to/from results cache
        )

    @doc(
        summary="""
            Plot an unrooted neighbour-joining tree, computed from pairwise distances
            between samples using biallelic SNP genotypes.
        """,
        extended_summary="""
            The tree is displayed as an unrooted tree using the equal angles layout.
        """,
        parameters=dict(
            center_x="X coordinate where plotting is centered.",
            center_y="Y coordinate where plotting is centered.",
            arc_start="Angle where tree layout begins.",
            arc_stop="Angle where tree layout ends.",
            edge_legend="Show legend entries for the different edge (line) colors.",
            leaf_legend="Show legend entries for the different leaf node (scatter) colors and symbols.",
            legend_sizing="Controls sizing of items in legends, either 'trace' or 'constant'.",
        ),
    )
    def plot_njt(
        self,
        region: base_params.regions,
        n_snps: base_params.n_snps,
        color: plotly_params.color = None,
        symbol: plotly_params.symbol = None,
        metric: diplotype_distance_params.distance_metric = "cityblock",
        distance_sort: Optional[tree_params.distance_sort] = None,
        count_sort: Optional[tree_params.count_sort] = None,
        center_x=0,
        center_y=0,
        arc_start=0,
        arc_stop=2 * math.pi,
        width: plotly_params.fig_width = 800,
        height: plotly_params.fig_height = 600,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        render_mode: plotly_params.render_mode = "svg",
        title: plotly_params.title = True,
        title_font_size: plotly_params.title_font_size = 14,
        line_width: plotly_params.line_width = 0.5,
        marker_size: plotly_params.marker_size = 5,
        color_discrete_sequence: plotly_params.color_discrete_sequence = None,
        color_discrete_map: plotly_params.color_discrete_map = None,
        category_orders: plotly_params.category_order = None,
        edge_legend: bool = False,
        leaf_legend: bool = True,
        legend_sizing: plotly_params.legend_sizing = "constant",
        thin_offset: base_params.thin_offset = 0,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_indices: Optional[base_params.sample_indices] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        site_class: Optional[base_params.site_class] = None,
        min_minor_ac: Optional[
            base_params.min_minor_ac
        ] = pca_params.min_minor_ac_default,
        max_missing_an: Optional[
            base_params.max_missing_an
        ] = pca_params.max_missing_an_default,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
        random_seed: base_params.random_seed = 42,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.chunks_default,
    ) -> plotly_params.figure:
        from biotite.sequence.phylo import neighbor_joining  # type: ignore
        from scipy.spatial.distance import squareform  # type: ignore

        # Normalise params.
        if count_sort is None and distance_sort is None:
            count_sort = True
            distance_sort = False

        # Compute pairwise distances.
        dist, samples, n_snps_used = self.biallelic_diplotype_pairwise_distances(
            region=region,
            n_snps=n_snps,
            metric=metric,
            thin_offset=thin_offset,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_indices=sample_indices,
            site_mask=site_mask,
            site_class=site_class,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            inline_array=inline_array,
            chunks=chunks,
        )

        # Construct tree.
        with self._spinner("Construct neighbour-joining tree"):
            dist_sq = squareform(dist)
            tree = neighbor_joining(dist_sq)

        # Load sample metadata.
        df_samples = self.sample_metadata(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_indices=sample_indices,
        )
        # Ensure alignment with pairwise distances.
        df_samples = df_samples.set_index("sample_id").loc[samples].reset_index()

        # Normalise color and symbol parameters.
        symbol_prepped = self._setup_sample_symbol(
            data=df_samples,
            symbol=symbol,
        )
        del symbol
        (
            color_prepped,
            color_discrete_map_prepped,
            category_orders_prepped,
        ) = self._setup_sample_colors_plotly(
            data=df_samples,
            color=color,
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=color_discrete_sequence,
            category_orders=category_orders,
        )
        del color
        del color_discrete_map
        del color_discrete_sequence

        # Extract data for leaf colors.
        if color_prepped is not None:
            leaf_colors = df_samples[color_prepped]
        else:
            leaf_colors = None

        # Lay out the tree.
        _, df_leaf_nodes, df_edges = unrooted_tree_layout_equal_angle(
            tree=tree,
            color=color_prepped,
            leaf_colors=leaf_colors,
            center_x=center_x,
            center_y=center_y,
            arc_start=arc_start,
            arc_stop=arc_stop,
            count_sort=count_sort,
            distance_sort=distance_sort,
        )

        # Join leaf data with sample metadata.
        df_leaf_data = df_leaf_nodes[["x", "y"]].join(df_samples, how="inner")
        df_leaf_data.sort_index(inplace=True)

        # Force default color for lines.
        if color_discrete_map_prepped is not None:
            color_discrete_map_prepped[""] = "gray"

        # Construct plot title.
        if title is True:
            title_lines = []
            if sample_sets is not None:
                title_lines.append(f"Sample sets: {sample_sets}")
            if sample_query is not None:
                title_lines.append(f"Sample query: {sample_query}")
            title_lines.append(f"Genomic region: {region} ({n_snps_used:,} SNPs)")
            title = "<br>".join(title_lines)

        # Start building the figure.
        fig1 = px.line(
            data_frame=df_edges,
            x="x",
            y="y",
            color=color_prepped,
            hover_name=color_prepped,
            hover_data=None,
            color_discrete_map=color_discrete_map_prepped,
            category_orders=category_orders_prepped,
            render_mode=render_mode,
        )
        fig1.update_traces(
            showlegend=edge_legend,
        )

        # Configure hover data.
        hover_data = self._setup_sample_hover_data_plotly(
            color=color_prepped, symbol=symbol_prepped
        )

        # Add scatter plot to draw the leaves.
        fig2 = px.scatter(
            data_frame=df_leaf_data,
            x="x",
            y="y",
            color=color_prepped,
            symbol=symbol_prepped,
            color_discrete_map=color_discrete_map_prepped,
            category_orders=category_orders_prepped,
            render_mode=render_mode,
            hover_name="sample_id",
            hover_data=hover_data,
        )
        fig2.update_traces(
            showlegend=leaf_legend,
        )

        fig = go.Figure()
        fig.add_traces(list(fig1.select_traces()))
        fig.add_traces(list(fig2.select_traces()))

        # Style lines and markers.
        line_props = dict(width=line_width)
        marker_props = dict(size=marker_size)
        fig.update_traces(line=line_props, marker=marker_props)

        # Tidy up.
        fig.update_layout(
            title=title,
            width=width,
            height=height,
            template="simple_white",
            title_font=dict(
                size=title_font_size,
            ),
            legend=dict(itemsizing=legend_sizing, tracegroupgap=0),
        )

        # Style axes.
        fig.update_xaxes(
            title=None,
            mirror=False,
            showgrid=False,
            showline=False,
            showticklabels=False,
            ticks="",
        )
        fig.update_yaxes(
            title=None,
            mirror=False,
            showgrid=False,
            showline=False,
            showticklabels=False,
            ticks="",
            # N.B., this is important, as it prevents distortion of the tree.
            # See also https://plotly.com/python/axes/#fixed-ratio-axes
            scaleanchor="x",
            scaleratio=1,
        )

        if show:  # pragma: no cover
            fig.show(renderer=renderer)
            return None
        else:
            return fig


def unrooted_tree_layout_equal_angle(
    *,
    tree,
    color,
    leaf_colors,
    center_x,
    center_y,
    arc_start,
    arc_stop,
    distance_sort,
    count_sort,
):
    # Set up outputs.
    internal_nodes = []
    leaf_nodes = []
    edges = []

    # Begin recursion.
    _unrooted_tree_layout_equal_angle(
        tree_node=tree.root,
        leaf_colors=leaf_colors,
        x=center_x,
        y=center_y,
        arc_start=arc_start,
        arc_stop=arc_stop,
        distance_sort=distance_sort,
        count_sort=count_sort,
        internal_nodes=internal_nodes,
        leaf_nodes=leaf_nodes,
        edges=edges,
    )

    # Load results into dataframes.
    df_internal_nodes = pd.DataFrame.from_records(internal_nodes, columns=["x", "y"])
    df_leaf_nodes = pd.DataFrame.from_records(
        leaf_nodes, columns=["x", "y", "index", color]
    ).set_index("index")
    df_edges = pd.DataFrame.from_records(edges, columns=["x", "y", color])

    return df_internal_nodes, df_leaf_nodes, df_edges


def _unrooted_tree_layout_equal_angle(
    tree_node,
    leaf_colors,
    x,
    y,
    arc_start,
    arc_stop,
    distance_sort,
    count_sort,
    internal_nodes,
    leaf_nodes,
    edges,
):
    if tree_node.children:
        # Store internal node coordinates.
        internal_nodes.append([x, y])

        # Count leaves (descendants).
        leaf_count = tree_node.get_leaf_count()

        # Sort the subtrees.
        if distance_sort:
            children = sorted(tree_node.children, key=lambda c: c.distance)
        elif count_sort:
            children = sorted(tree_node.children, key=lambda c: c.get_leaf_count())
        else:
            children = tree_node.children

        # Iterate over children, dividing up the current arc into
        # segments of size proportional to the number of leaves in
        # the subtree.
        arc_size = arc_stop - arc_start
        child_arc_start = arc_start
        for child in children:
            # Define a segment of the arc for this child.
            child_arc_size = arc_size * child.get_leaf_count() / leaf_count
            child_arc_stop = child_arc_start + child_arc_size

            # Define the angle at which this child will be drawn.
            child_angle = child_arc_start + child_arc_size / 2

            # Access the distance at which to draw this child.
            distance = child.distance

            # Now use trigonometry to calculate coordinates for this child.
            child_x = x + distance * math.sin(child_angle)
            child_y = y + distance * math.cos(child_angle)

            # Figure out edge colors.
            edge_color = ""
            if leaf_colors is not None:
                edge_colors = set(
                    leaf_colors.iloc[[leaf.index for leaf in child.get_leaves()]]
                )
                if len(edge_colors) == 1:
                    edge_color = edge_colors.pop()

            # Add edge.
            # edges.append([x, child_x, y, child_y, edge_color])
            edges.append([x, y, edge_color])
            edges.append([child_x, child_y, edge_color])
            edges.append([None, None, edge_color])

            # Recurse to draw the child.
            _unrooted_tree_layout_equal_angle(
                tree_node=child,
                leaf_colors=leaf_colors,
                x=child_x,
                y=child_y,
                internal_nodes=internal_nodes,
                leaf_nodes=leaf_nodes,
                edges=edges,
                arc_start=child_arc_start,
                arc_stop=child_arc_stop,
                count_sort=count_sort,
                distance_sort=distance_sort,
            )

            # Update loop variables ready for the next iteration.
            child_arc_start = child_arc_stop

    else:
        leaf_color = ""
        if leaf_colors is not None:
            leaf_color = leaf_colors.iloc[tree_node.index]
        leaf_nodes.append([x, y, tree_node.index, leaf_color])


@numba.njit("Tuple((int8, int64))(int8[:], int8)")
def _cn_mode_1d(a, vmax):
    # setup intermediates
    m = a.shape[0]
    counts = np.zeros(vmax + 1, dtype=numba.int64)

    # initialise return values
    mode = numba.int8(-1)
    mode_count = numba.int64(0)

    # iterate over array values, keeping track of counts
    for i in range(m):
        v = a[i]
        if 0 <= v <= vmax:
            c = counts[v]
            c += 1
            counts[v] = c
            if c > mode_count:
                mode = v
                mode_count = c
            elif c == mode_count and v < mode:
                # consistency with scipy.stats, break ties by taking lower value
                mode = v

    return mode, mode_count


@numba.njit("Tuple((int8[:], int64[:]))(int8[:, :], int8)")
def _cn_mode(a, vmax):
    # setup intermediates
    n = a.shape[1]

    # setup outputs
    modes = np.zeros(n, dtype=numba.int8)
    counts = np.zeros(n, dtype=numba.int64)

    # iterate over columns, computing modes
    for j in range(a.shape[1]):
        mode, count = _cn_mode_1d(a[:, j], vmax)
        modes[j] = mode
        counts[j] = count

    return modes, counts
