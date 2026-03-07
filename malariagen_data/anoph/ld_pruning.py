from typing import Optional

import allel  # type: ignore
import numpy as np
import xarray as xr

from numpydoc_decorator import doc  # type: ignore

from ..util import CacheMiss, _check_types, _dask_compress_dataset
from . import base_params, ld_pruning_params, pca_params
from .snp_data import AnophelesSnpData


class AnophelesLdPruning(
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

    @_check_types
    @doc(
        summary="""
            Perform LD pruning on biallelic SNP data.
        """,
        extended_summary="""
            This function performs linkage disequilibrium (LD) pruning
            on biallelic SNPs. It uses the Rogers and Huff r² statistic
            in a sliding window approach to iteratively remove SNPs that
            are in LD. The result is a set of approximately independent
            SNPs suitable for analyses that assume linkage equilibrium,
            such as ADMIXTURE, PCA, or GWAS.
        """,
        returns="""
            Dataset of biallelic SNP calls after LD pruning, with the
            same variables as returned by ``biallelic_snp_calls()``.
        """,
        notes="""
            This computation may take some time to run, depending on your
            computing environment. Results of this computation will be cached
            and re-used if the ``results_cache`` parameter was set when
            instantiating the API client.
        """,
    )
    def ld_prune(
        self,
        region: base_params.regions,
        n_snps: Optional[base_params.n_snps] = None,
        thin_offset: base_params.thin_offset = 0,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        sample_indices: Optional[base_params.sample_indices] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
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
        r2_threshold: ld_pruning_params.r2_threshold = ld_pruning_params.r2_threshold_default,
        window_size: ld_pruning_params.window_size = ld_pruning_params.window_size_default,
        window_step: ld_pruning_params.window_step = ld_pruning_params.window_step_default,
        n_iter: ld_pruning_params.n_iter = ld_pruning_params.n_iter_default,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
    ) -> xr.Dataset:
        # Change this name if you ever change the behaviour of this function,
        # to invalidate any previously cached data.
        name = "ld_prune_v1"

        # Check that either sample_query xor sample_indices are provided.
        base_params._validate_sample_selection_params(
            sample_query=sample_query, sample_indices=sample_indices
        )

        ## Normalize params for consistent hash value.

        # Note: `_prep_sample_selection_cache_params` converts `sample_query`
        # and `sample_query_options` into `sample_indices`.
        # So `sample_query` and `sample_query_options` should not be used
        # beyond this point. (`sample_indices` should be used instead.)
        (
            prepared_sample_sets,
            prepared_sample_indices,
        ) = self._prep_sample_selection_cache_params(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            sample_indices=sample_indices,
        )
        prepared_region = self._prep_region_cache_param(region=region)
        prepared_site_mask = self._prep_optional_site_mask_param(site_mask=site_mask)

        # Delete original parameters to prevent accidental use.
        del sample_sets
        del sample_indices
        del sample_query
        del sample_query_options
        del region
        del site_mask

        params = dict(
            region=prepared_region,
            n_snps=n_snps,
            thin_offset=thin_offset,
            sample_sets=prepared_sample_sets,
            sample_indices=prepared_sample_indices,
            site_mask=prepared_site_mask,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            r2_threshold=r2_threshold,
            window_size=window_size,
            window_step=window_step,
            n_iter=n_iter,
        )

        # Try to retrieve results from the cache.
        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._ld_prune(inline_array=inline_array, chunks=chunks, **params)
            self.results_cache_set(name=name, params=params, results=results)

        # Unpack the cached pruning mask.
        loc_pruned = results["loc_pruned"]

        # Reload the biallelic SNP calls dataset (lazy/dask).
        ds = self.biallelic_snp_calls(
            region=prepared_region,
            sample_sets=prepared_sample_sets,
            sample_indices=prepared_sample_indices,
            site_mask=prepared_site_mask,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            n_snps=n_snps,
            thin_offset=thin_offset,
            inline_array=inline_array,
            chunks=chunks,
        )

        # Apply the LD pruning mask.
        ds_pruned = _dask_compress_dataset(ds, indexer=loc_pruned, dim="variants")

        return ds_pruned

    def _ld_prune(
        self,
        *,
        region,
        n_snps,
        thin_offset,
        sample_sets,
        sample_indices,
        site_mask,
        min_minor_ac,
        max_missing_an,
        cohort_size,
        min_cohort_size,
        max_cohort_size,
        random_seed,
        r2_threshold,
        window_size,
        window_step,
        n_iter,
        inline_array,
        chunks,
    ):
        # Load biallelic SNP calls.
        ds_snps = self.biallelic_snp_calls(
            region=region,
            sample_sets=sample_sets,
            sample_indices=sample_indices,
            site_mask=site_mask,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            n_snps=n_snps,
            thin_offset=thin_offset,
            random_seed=random_seed,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            inline_array=inline_array,
            chunks=chunks,
        )

        # Compute diplotype array (ref allele counts per individual).
        with self._dask_progress("Computing diplotypes for LD pruning"):
            gt = allel.GenotypeDaskArray(ds_snps["call_genotype"].data)
            gn = gt.to_n_ref(fill=-127).compute()

        # Remove sites with no variation for numerical stability.
        loc_var = np.any(gn != gn[:, 0, np.newaxis], axis=1)
        gn_var = gn[loc_var]

        # Iterative LD pruning using Rogers-Huff r².
        loc_unlinked = np.ones(gn_var.shape[0], dtype=bool)

        with self._spinner(f"LD pruning (r²={r2_threshold}, window={window_size})"):
            for i in range(n_iter):
                # Compute LD on current set of unlinked SNPs.
                gn_current = gn_var[loc_unlinked]
                if gn_current.shape[0] < 2:
                    break

                loc_iter = allel.locate_unlinked(
                    gn_current,
                    size=window_size,
                    step=window_step,
                    threshold=r2_threshold,
                )

                # Map back to full index.
                unlinked_indices = np.where(loc_unlinked)[0]
                removed_indices = unlinked_indices[~loc_iter]
                loc_unlinked[removed_indices] = False

                n_after = np.sum(loc_unlinked)
                if n_after == np.sum(loc_iter):
                    # No more SNPs removed, converged.
                    break

        # Map loc_unlinked back to full dataset (accounting for loc_var).
        var_indices = np.where(loc_var)[0]
        pruned_indices = var_indices[loc_unlinked]
        loc_pruned = np.zeros(ds_snps.sizes["variants"], dtype=bool)
        loc_pruned[pruned_indices] = True

        return dict(loc_pruned=loc_pruned)
