from typing import Optional, Tuple

import allel  # type: ignore
import numpy as np
from numpydoc_decorator import doc  # type: ignore

from ..util import (
    CacheMiss,
    check_types,
    multiallelic_diplotype_pdist,
    multiallelic_diplotype_mean_sqeuclidean,
    multiallelic_diplotype_mean_cityblock,
)
from ..plotly_dendrogram import plot_dendrogram
from . import base_params, plotly_params, tree_params, dipclust_params
from .base_params import DEFAULT
from .snp_data import AnophelesSnpData


class AnophelesDipClustAnalysis(
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

    @check_types
    @doc(
        summary="Hierarchically cluster diplotypes in region and produce an interactive plot.",
        parameters=dict(
            leaf_y="Y coordinate at which to plot the leaf markers.",
        ),
    )
    def plot_diplotype_clustering(
        self,
        region: base_params.regions,
        site_mask: base_params.site_mask = DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        random_seed: base_params.random_seed = 42,
        color: plotly_params.color = None,
        symbol: plotly_params.symbol = None,
        linkage_method: dipclust_params.linkage_method = dipclust_params.linkage_method_default,
        distance_metric: dipclust_params.distance_metric = dipclust_params.distance_metric_default,
        count_sort: Optional[tree_params.count_sort] = None,
        distance_sort: Optional[tree_params.distance_sort] = None,
        title: plotly_params.title = True,
        title_font_size: plotly_params.title_font_size = 14,
        width: plotly_params.width = None,
        height: plotly_params.height = 500,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        render_mode: plotly_params.render_mode = "svg",
        leaf_y: int = 0,
        marker_size: plotly_params.marker_size = 5,
        line_width: plotly_params.line_width = 0.5,
        line_color: plotly_params.line_color = "black",
        color_discrete_sequence: plotly_params.color_discrete_sequence = None,
        color_discrete_map: plotly_params.color_discrete_map = None,
        category_orders: plotly_params.category_order = None,
        legend_sizing: plotly_params.legend_sizing = "constant",
    ) -> plotly_params.figure:
        import sys

        debug = self._log.debug

        # Normalise params.
        if count_sort is None and distance_sort is None:
            count_sort = True
            distance_sort = False

        # This is needed to avoid RecursionError on some haplotype clustering analyses
        # with larger numbers of haplotypes.
        sys.setrecursionlimit(10_000)

        debug("load sample metadata")
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )

        dist, gt_samples, n_snps_used = self.diplotype_pairwise_distances(
            region=region,
            site_mask=site_mask,
            sample_sets=sample_sets,
            sample_query=sample_query,
            cohort_size=cohort_size,
            distance_metric=distance_metric,
            random_seed=random_seed,
        )

        # Align sample metadata with genotypes.
        df_samples = (
            df_samples.set_index("sample_id").loc[gt_samples.tolist()].reset_index()
        )

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
            fig = plot_dendrogram(
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
                leaf_data=df_samples,
                leaf_hover_name="sample_id",
                leaf_hover_data=hover_data,
                leaf_color=color_prepped,
                leaf_symbol=symbol_prepped,
                leaf_y=leaf_y,
                leaf_color_discrete_map=color_discrete_map_prepped,
                leaf_category_orders=category_orders_prepped,
                template="simple_white",
                y_axis_title=f"Distance ({distance_metric})",
                y_axis_buffer=0.1,
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

    def diplotype_pairwise_distances(
        self,
        region: base_params.regions,
        site_mask: base_params.site_mask = DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        site_class: Optional[base_params.site_class] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        distance_metric: dipclust_params.distance_metric = dipclust_params.distance_metric_default,
        random_seed: base_params.random_seed = 42,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "diplotype_pairwise_distances_v1"

        # Normalize params for consistent hash value.
        sample_sets_prepped = self._prep_sample_sets_param(sample_sets=sample_sets)
        region_prepped = self._prep_region_cache_param(region=region)
        params = dict(
            region=region_prepped,
            site_mask=site_mask,
            sample_sets=sample_sets_prepped,
            sample_query=sample_query,
            site_class=site_class,
            cohort_size=cohort_size,
            distance_metric=distance_metric,
            random_seed=random_seed,
        )

        # Try to retrieve results from the cache.
        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._diplotype_pairwise_distances(**params)
            self.results_cache_set(name=name, params=params, results=results)

        # Unpack results")
        dist: np.ndarray = results["dist"]
        gt_samples: np.ndarray = results["gt_samples"]
        n_snps: int = int(results["n_snps"][()])  # ensure scalar

        return dist, gt_samples, n_snps

    def _diplotype_pairwise_distances(
        self,
        *,
        region,
        site_mask,
        sample_sets,
        sample_query,
        site_class,
        cohort_size,
        distance_metric,
        random_seed,
    ):
        if distance_metric == "cityblock":
            metric = multiallelic_diplotype_mean_cityblock
        elif distance_metric == "euclidean":
            metric = multiallelic_diplotype_mean_sqeuclidean

        # Load haplotypes.
        ds_snps = self.snp_calls(
            region=region,
            sample_query=sample_query,
            sample_sets=sample_sets,
            site_mask=site_mask,
            site_class=site_class,
            cohort_size=cohort_size,
            random_seed=random_seed,
        )

        with self._dask_progress(desc="Load genotypes"):
            gt = ds_snps["call_genotype"].data.compute()

        with self._spinner(
            desc="Compute allele counts and remove non-segregating sites"
        ):
            # Compute allele count, remove non-segregating sites.
            ac = allel.GenotypeArray(gt).count_alleles(max_allele=3)
            gt_seg = gt.compress(ac.is_segregating(), axis=0)
            ac_seg = allel.GenotypeArray(gt_seg).to_allele_counts(max_allele=3)
            X = np.ascontiguousarray(np.swapaxes(ac_seg.values, 0, 1))

        # Compute pairwise distances.
        with self._spinner(desc="Compute pairwise distances"):
            dist = multiallelic_diplotype_pdist(X, metric=metric)

        # Extract IDs of samples. Convert to "U" dtype here
        # to allow these to be saved to the results cache.
        gt_samples = ds_snps["sample_id"].values.astype("U")

        return dict(
            dist=dist,
            gt_samples=gt_samples,
            n_snps=np.array(gt_seg.shape[0]),
        )
