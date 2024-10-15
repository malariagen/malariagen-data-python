from typing import Optional, Tuple
import numba  # type: ignore
import numpy as np
from numpydoc_decorator import doc  # type: ignore
from .snp_data import AnophelesSnpData
from . import base_params, distance_params
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
        parameters=dict(
            metric="Distance metric, one of 'cityblock', 'euclidean' or 'sqeuclidean'.",
        ),
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
            Compute pairwise distances between samples using biallelic SNP genotypes.
        """,
        parameters=dict(
            metric="Distance metric, one of 'cityblock', 'euclidean' or 'sqeuclidean'.",
        ),
    )
    def njt(
        self,
        region: base_params.regions,
        n_snps: base_params.n_snps,
        algorithm: distance_params.nj_algorithm = "rapid",
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
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "njt"

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
        import anjl
        from scipy.spatial.distance import squareform

        # Compute diplotypes.
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

        # Progress doesn't mix well with debug logging.
        show_progress = self._show_progress and not self._debug
        if show_progress:
            progress = self._tqdm_class
            progress_options = dict(desc="Compute neighbour-joining", leave=False)
        else:
            progress = None
            progress_options = dict()

        if algorithm == "rapid":
            Z = anjl.rapid_nj(D=D, progress=progress, progress_options=progress_options)
        else:
            Z = anjl.canonical_nj(
                D=D, progress=progress, progress_options=progress_options
            )

        return dict(
            Z=Z,
            samples=samples,
            n_snps=np.array(
                n_snps
            ),  # ensure consistent behaviour to/from results cache
        )
