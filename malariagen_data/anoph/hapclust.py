from typing import Optional, Tuple

import allel  # type: ignore
import numpy as np
import pandas as pd
from numpydoc_decorator import doc  # type: ignore

from ..util import CacheMiss, check_types, pdist_abs_hamming
from ..plotly_dendrogram import plot_dendrogram
from . import (
    base_params,
    plotly_params,
    tree_params,
    hap_params,
    clustering_params,
    hapclust_params,
)
from .snp_data import AnophelesSnpData
from .hap_data import AnophelesHapData


class AnophelesHapClustAnalysis(AnophelesHapData, AnophelesSnpData):
    def __init__(
        self,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

    @check_types
    @doc(
        summary="""
            Hierarchically cluster haplotypes in region and produce an interactive plot.
        """,
    )
    def plot_haplotype_clustering(
        self,
        region: base_params.regions,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        random_seed: base_params.random_seed = 42,
        color: plotly_params.color = None,
        symbol: plotly_params.symbol = None,
        linkage_method: hapclust_params.linkage_method = hapclust_params.linkage_method_default,
        count_sort: Optional[tree_params.count_sort] = None,
        distance_sort: Optional[tree_params.distance_sort] = None,
        title: plotly_params.title = True,
        title_font_size: plotly_params.title_font_size = 14,
        width: plotly_params.fig_width = None,
        height: plotly_params.fig_height = 500,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        render_mode: plotly_params.render_mode = "svg",
        leaf_y: clustering_params.leaf_y = 0,
        marker_size: plotly_params.marker_size = 5,
        line_width: plotly_params.line_width = 0.5,
        line_color: plotly_params.line_color = "black",
        color_discrete_sequence: plotly_params.color_discrete_sequence = None,
        color_discrete_map: plotly_params.color_discrete_map = None,
        category_orders: plotly_params.category_order = None,
        legend_sizing: plotly_params.legend_sizing = "constant",
    ) -> plotly_params.figure:
        import sys

        # Normalise params.
        if count_sort is None and distance_sort is None:
            count_sort = True
            distance_sort = False

        # This is needed to avoid RecursionError on some haplotype clustering analyses
        # with larger numbers of haplotypes.
        sys.setrecursionlimit(10_000)

        # Load sample metadata.
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )

        # Compute pairwise distances.
        dist, phased_samples, n_snps_used = self.haplotype_pairwise_distances(
            region=region,
            analysis=analysis,
            sample_sets=sample_sets,
            sample_query=sample_query,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        # Align sample metadata with haplotypes.
        df_samples_phased = (
            df_samples.set_index("sample_id").loc[phased_samples.tolist()].reset_index()
        )

        # Normalise color and symbol parameters.
        symbol_prepped = self._setup_sample_symbol(
            data=df_samples_phased,
            symbol=symbol,
        )
        del symbol
        (
            color_prepped,
            color_discrete_map_prepped,
            category_orders_prepped,
        ) = self._setup_sample_colors_plotly(
            data=df_samples_phased,
            color=color,
            color_discrete_map=color_discrete_map,
            color_discrete_sequence=color_discrete_sequence,
            category_orders=category_orders,
        )
        del color
        del color_discrete_map
        del color_discrete_sequence

        # Repeat the dataframe so there is one row of metadata for each haplotype.
        df_haps = pd.DataFrame(np.repeat(df_samples_phased.values, 2, axis=0))
        df_haps.columns = df_samples_phased.columns

        # Configure hover data.
        hover_data = self._setup_sample_hover_data_plotly(
            color=color_prepped, symbol=symbol_prepped
        )

        # Construct plot title.
        if title is True:
            title_lines = []
            if sample_sets is not None:
                title_lines.append(f"Sample sets: {sample_sets}")
            if sample_query is not None:
                title_lines.append(f"Sample query: {sample_query}")
            title_lines.append(f"Genomic region: {region} ({n_snps_used:,} SNPs)")
            title = "<br>".join(title_lines)

        # Create the plot.
        with self._spinner("Plot dendrogram"):
            fig, _ = plot_dendrogram(
                dist=dist,
                linkage_method=linkage_method,
                count_sort=count_sort,
                distance_sort=distance_sort,
                render_mode=render_mode,
                width=width,
                height=height,
                title=title,
                line_width=line_width,
                line_color=line_color,
                marker_size=marker_size,
                leaf_data=df_haps,
                leaf_hover_name="sample_id",
                leaf_hover_data=hover_data,
                leaf_color=color_prepped,
                leaf_symbol=symbol_prepped,
                leaf_y=leaf_y,
                leaf_color_discrete_map=color_discrete_map_prepped,
                leaf_category_orders=category_orders_prepped,
                template="simple_white",
                y_axis_title="Distance (no. SNPs)",
                y_axis_buffer=1,
            )

        # Tidy up.
        fig.update_layout(
            title_font=dict(
                size=title_font_size,
            ),
            legend=dict(itemsizing=legend_sizing, tracegroupgap=0),
        )

        if show:  # pragma: no cover
            fig.show(renderer=renderer)
            return None
        else:
            return fig

    @doc(
        summary="""
            Compute pairwise distances between haplotypes.
        """,
        returns=dict(
            dist="Pairwise distance.",
            phased_samples="Sample identifiers for haplotypes.",
            n_snps="Number of SNPs used.",
        ),
    )
    def haplotype_pairwise_distances(
        self,
        region: base_params.regions,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        random_seed: base_params.random_seed = 42,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "haplotype_pairwise_distances"

        # Normalize params for consistent hash value.
        sample_sets_prepped = self._prep_sample_sets_param(sample_sets=sample_sets)
        region_prepped = self._prep_region_cache_param(region=region)
        params = dict(
            region=region_prepped,
            analysis=analysis,
            sample_sets=sample_sets_prepped,
            sample_query=sample_query,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        # Try to retrieve results from the cache.
        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._haplotype_pairwise_distances(**params)
            self.results_cache_set(name=name, params=params, results=results)

        # Unpack results")
        dist: np.ndarray = results["dist"]
        phased_samples: np.ndarray = results["phased_samples"]
        n_snps: int = int(results["n_snps"][()])  # ensure scalar

        return dist, phased_samples, n_snps

    def _haplotype_pairwise_distances(
        self,
        *,
        region,
        analysis,
        sample_sets,
        sample_query,
        cohort_size,
        random_seed,
    ):
        from scipy.spatial.distance import squareform  # type: ignore

        # Load haplotypes.
        ds_haps = self.haplotypes(
            region=region,
            analysis=analysis,
            sample_query=sample_query,
            sample_sets=sample_sets,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )
        gt = allel.GenotypeDaskArray(ds_haps["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes"):
            ht = gt.to_haplotypes().compute().values

        # Compute allele count, remove non-segregating sites.
        ac = allel.HaplotypeArray(ht).count_alleles(max_allele=1)
        ht_seg = ht[ac.is_segregating()]

        # Transpose memory layout for faster hamming distance calculations.
        ht_t = np.ascontiguousarray(ht_seg.T)

        # Compute pairwise distances.
        with self._spinner(desc="Compute pairwise distances"):
            dist_sq = pdist_abs_hamming(ht_t)
        dist = squareform(dist_sq)

        # Extract IDs of phased samples. Convert to "U" dtype here
        # to allow these to be saved to the results cache.
        phased_samples = ds_haps["sample_id"].values.astype("U")

        return dict(
            dist=dist,
            phased_samples=phased_samples,
            n_snps=np.array(ht.shape[0]),
        )
