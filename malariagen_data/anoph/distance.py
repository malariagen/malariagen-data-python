# Standard library imports.
from typing import Optional, Tuple
import math

# Third-party library imports.
import numba  # type: ignore
import numpy as np
from numpydoc_decorator import doc  # type: ignore

# Internal imports.
from .snp_data import AnophelesSnpData
from . import base_params, distance_params, plotly_params, pca_params, tree_params
from ..util import square_to_condensed, check_types, CacheMiss


@numba.njit(parallel=True)
def biallelic_diplotype_pdist(X, distfun):
    n_samples = X.shape[0]
    n_pairs = (n_samples * (n_samples - 1)) // 2
    out = np.zeros(n_pairs, dtype=np.float32)

    # Loop over samples, first in pair.
    for i in range(n_samples):
        x = X[i, :]

        # Loop over observations again, second in pair.
        for j in numba.prange(i + 1, n_samples):
            y = X[j, :]

            # Compute distance for the current pair.
            d = distfun(x, y)

            # Store result for the current pair.
            k = square_to_condensed(i, j, n_samples)
            out[k] = d

    return out


@numba.njit
def biallelic_diplotype_cityblock(x, y):
    n_sites = x.shape[0]
    distance = np.float32(0)

    # Loop over sites.
    for i in range(n_sites):
        # Compute cityblock distance (absolute difference).
        d = np.fabs(x[i] - y[i])

        # Accumulate distance for the current pair.
        distance += d

    return distance


@numba.njit
def biallelic_diplotype_sqeuclidean(x, y):
    n_sites = x.shape[0]
    distance = np.float32(0)

    # Loop over sites.
    for i in range(n_sites):
        # Compute squared euclidean distance.
        d = (x[i] - y[i]) ** 2

        # Accumulate distance for the current pair.
        distance += d

    return distance


@numba.njit
def biallelic_diplotype_euclidean(x, y):
    return np.sqrt(biallelic_diplotype_sqeuclidean(x, y))


class AnophelesDistanceAnalysis(AnophelesSnpData):
    def __init__(self, **kwargs):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

    @check_types
    @doc(
        summary="""
            Compute pairwise distances between samples using biallelic SNP genotypes.
        """,
        returns=("dist", "samples", "n_snps_used"),
    )
    def biallelic_diplotype_pairwise_distances(
        self,
        region: base_params.regions,
        n_snps: base_params.n_snps,
        metric: distance_params.distance_metric = "cityblock",
        thin_offset: base_params.thin_offset = 0,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
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
        chunks: base_params.chunks = base_params.native_chunks,
    ) -> Tuple[
        distance_params.dist, distance_params.samples, distance_params.n_snps_used
    ]:
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
            sample_query_options=sample_query_options,
            sample_indices=sample_indices,
        )
        region_prepped = self._prep_region_cache_param(region=region)
        site_mask_prepped = self._prep_optional_site_mask_param(site_mask=site_mask)
        del sample_sets
        del sample_query
        del sample_query_options
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

    @check_types
    @doc(
        summary="""
            Construct a neighbour-joining tree between samples using biallelic SNP genotypes.
        """,
        returns=("Z", "samples", "n_snps_used"),
    )
    def njt(
        self,
        region: base_params.regions,
        n_snps: base_params.n_snps,
        algorithm: distance_params.nj_algorithm = distance_params.default_nj_algorithm,
        metric: distance_params.distance_metric = distance_params.default_distance_metric,
        thin_offset: base_params.thin_offset = 0,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
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
        chunks: base_params.chunks = base_params.native_chunks,
    ) -> Tuple[distance_params.Z, distance_params.samples, distance_params.n_snps_used]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "njt_v1"

        # Normalize params for consistent hash value.
        (
            sample_sets_prepped,
            sample_indices_prepped,
        ) = self._prep_sample_selection_cache_params(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            sample_indices=sample_indices,
        )
        region_prepped = self._prep_region_cache_param(region=region)
        site_mask_prepped = self._prep_optional_site_mask_param(site_mask=site_mask)
        del sample_sets
        del sample_query
        del sample_query_options
        del sample_indices
        del region
        del site_mask
        params = dict(
            region=region_prepped,
            n_snps=n_snps,
            algorithm=algorithm,
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
            results = self._njt(inline_array=inline_array, chunks=chunks, **params)
            self.results_cache_set(name=name, params=params, results=results)

        # Unpack results.
        Z: np.ndarray = results["Z"]
        samples: np.ndarray = results["samples"]
        n_snps_used: int = int(results["n_snps"][()])  # ensure scalar

        return Z, samples, n_snps_used

    def _njt(
        self,
        region,
        n_snps,
        algorithm,
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
        # Only import anjl if needed, as it requires a couple of seconds to compile
        # functions.
        import anjl  # type: ignore
        from scipy.spatial.distance import squareform  # type: ignore

        # Compute pairwise distances.
        dist, samples, n_snps = self.biallelic_diplotype_pairwise_distances(
            region=region,
            n_snps=n_snps,
            metric=metric,
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
            thin_offset=thin_offset,
        )
        D = squareform(dist)

        # anjl supports passing in a progress bar function to get progress on the
        # neighbour-joining iterations.
        progress_options = dict(desc="Construct neighbour-joining tree", leave=False)

        # Decide which algorithm to use and run the neighbour-joining. The "dynamic"
        # algorithm is fastest.
        if algorithm == "dynamic":
            Z = anjl.dynamic_nj(
                D=D, progress=self._progress, progress_options=progress_options
            )
        elif algorithm == "rapid":
            Z = anjl.rapid_nj(
                D=D, progress=self._progress, progress_options=progress_options
            )
        else:
            Z = anjl.canonical_nj(
                D=D, progress=self._progress, progress_options=progress_options
            )

        return dict(
            Z=Z,
            samples=samples,
            n_snps=np.array(
                n_snps
            ),  # ensure consistent behaviour to/from results cache
        )

    @check_types
    @doc(
        summary="""
            Plot an unrooted neighbour-joining tree, computed from pairwise distances
            between samples using biallelic SNP genotypes.
        """,
        extended_summary="""
            The tree is displayed as an unrooted tree using the equal angles layout.
        """,
    )
    def plot_njt(
        self,
        region: base_params.regions,
        n_snps: base_params.n_snps,
        color: plotly_params.color = None,
        symbol: plotly_params.symbol = None,
        algorithm: distance_params.nj_algorithm = distance_params.default_nj_algorithm,
        metric: distance_params.distance_metric = distance_params.default_distance_metric,
        distance_sort: Optional[tree_params.distance_sort] = None,
        count_sort: Optional[tree_params.count_sort] = None,
        center_x: distance_params.center_x = 0,
        center_y: distance_params.center_y = 0,
        arc_start: distance_params.arc_start = 0,
        arc_stop: distance_params.arc_stop = 2 * math.pi,
        width: plotly_params.fig_width = 800,
        height: plotly_params.fig_height = 600,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        render_mode: plotly_params.render_mode = "auto",
        title: plotly_params.title = True,
        title_font_size: plotly_params.title_font_size = 14,
        line_width: plotly_params.line_width = 0.5,
        marker_size: plotly_params.marker_size = 5,
        color_discrete_sequence: plotly_params.color_discrete_sequence = None,
        color_discrete_map: plotly_params.color_discrete_map = None,
        category_orders: plotly_params.category_order = None,
        edge_legend: distance_params.edge_legend = False,
        leaf_legend: distance_params.leaf_legend = True,
        legend_sizing: plotly_params.legend_sizing = "constant",
        thin_offset: base_params.thin_offset = 0,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
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
        chunks: base_params.chunks = base_params.native_chunks,
    ) -> plotly_params.figure:
        # Only import anjl if needed, as it requires a couple of seconds to compile
        # functions.
        import anjl  # type: ignore

        # Normalise params.
        if count_sort is None and distance_sort is None:
            count_sort = True
            distance_sort = False

        # Compute neighbour-joining tree.
        Z, samples, n_snps_used = self.njt(
            region=region,
            n_snps=n_snps,
            algorithm=algorithm,
            metric=metric,
            thin_offset=thin_offset,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
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

        # Load sample metadata.
        df_samples = self.sample_metadata(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            sample_indices=sample_indices,
        )
        # Ensure alignment with the tree.
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

        # Construct plot title.
        if title is True:
            title_lines = []
            if sample_sets is not None:
                title_lines.append(f"Sample sets: {sample_sets}")
            if sample_query is not None:
                title_lines.append(f"Sample query: {sample_query}")
            title_lines.append(f"Genomic region: {region} ({n_snps_used:,} SNPs)")
            title = "<br>".join(title_lines)

        # Configure hover data.
        hover_data = self._setup_sample_hover_data_plotly(
            color=color_prepped, symbol=symbol_prepped
        )

        # Create the figure.
        fig = anjl.plot(
            Z=Z,
            leaf_data=df_samples,
            color=color_prepped,
            symbol=symbol_prepped,
            hover_name="sample_id",
            hover_data=hover_data,
            center_x=center_x,
            center_y=center_y,
            arc_start=arc_start,
            arc_stop=arc_stop,
            count_sort=count_sort,
            distance_sort=distance_sort,
            line_width=line_width,
            marker_size=marker_size,
            color_discrete_map=color_discrete_map_prepped,
            category_orders=category_orders_prepped,
            leaf_legend=leaf_legend,
            edge_legend=edge_legend,
            render_mode=render_mode,
            width=width,
            height=height,
            legend_sizing=legend_sizing,
            default_line_color="gray",
        )

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

        if show:  # pragma: no cover
            fig.show(renderer=renderer)
            return None
        else:
            return fig
