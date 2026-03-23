import warnings
from typing import Optional

import allel  # type: ignore
import numpy as np
import xarray as xr
from numpydoc_decorator import doc  # type: ignore

from ..util import CacheMiss, _check_types, _dask_compress_dataset
from . import base_params, ld_params, pca_params
from .snp_data import AnophelesSnpData


class AnophelesLdAnalysis(
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
            Return a dataset of biallelic SNPs that have been LD-pruned
            using the method of Rogers and Huff (2009).
        """,
        extended_summary="""
            This function first loads biallelic SNP calls for the given region
            and samples, then computes a diplotype matrix (number of reference
            alleles per call) and uses ``scikit-allel``'s
            ``locate_unlinked`` to identify a subset of SNPs in approximate
            linkage equilibrium. The pruning mask is cached so that repeated
            calls with the same parameters are fast.
        """,
        returns="""
            Dataset of biallelic SNP calls after LD pruning, with the same
            variables as returned by ``biallelic_snp_calls()``.
        """,
        notes="""
            This computation may take some time to run, depending on your
            computing environment. Results of this computation will be cached
            and re-used if the ``results_cache`` parameter was set when
            instantiating the API client.
        """,
    )
    def biallelic_snps_ld_pruned(
        self,
        region: base_params.regions,
        n_snps: Optional[base_params.n_snps] = None,
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
        size: ld_params.size = ld_params.size_default,
        step: ld_params.step = ld_params.step_default,
        threshold: ld_params.threshold = ld_params.threshold_default,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
    ) -> xr.Dataset:
        # Change this name if you ever change the behaviour of this function,
        # to invalidate any previously cached data.
        name = "biallelic_snps_ld_pruned_v1"

        # Check that either sample_query xor sample_indices are provided.
        base_params._validate_sample_selection_params(
            sample_query=sample_query, sample_indices=sample_indices
        )

        ## Normalize params for consistent hash value.

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
            site_class=site_class,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            random_seed=random_seed,
            size=size,
            step=step,
            threshold=threshold,
        )

        # Try to retrieve results from the cache.
        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._biallelic_snps_ld_pruned(
                inline_array=inline_array, chunks=chunks, **params
            )
            self.results_cache_set(name=name, params=params, results=results)

        # Unpack the cached mask.
        loc_unlinked = results["loc_unlinked"]

        # Reload the biallelic SNP calls dataset (lazy/dask).
        ds = self.biallelic_snp_calls(
            region=prepared_region,
            sample_sets=prepared_sample_sets,
            sample_indices=prepared_sample_indices,
            site_mask=prepared_site_mask,
            site_class=site_class,
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
        ds_pruned = _dask_compress_dataset(ds, indexer=loc_unlinked, dim="variants")

        return ds_pruned

    def _biallelic_snps_ld_pruned(
        self,
        *,
        region,
        n_snps,
        thin_offset,
        sample_sets,
        sample_indices,
        site_mask,
        site_class,
        min_minor_ac,
        max_missing_an,
        cohort_size,
        min_cohort_size,
        max_cohort_size,
        random_seed,
        size,
        step,
        threshold,
        inline_array,
        chunks,
    ):
        # Access biallelic SNP calls.
        ds = self.biallelic_snp_calls(
            region=region,
            sample_sets=sample_sets,
            sample_indices=sample_indices,
            site_mask=site_mask,
            site_class=site_class,
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

        # Estimate memory usage and warn the user.
        n_variants = ds.sizes["variants"]
        n_samples = ds.sizes["samples"]

        # Nothing to do if there are no variants.
        if n_variants == 0:
            return dict(loc_unlinked=np.empty(0, dtype=bool))

        estimated_mb = (n_variants * n_samples) / 1_000_000
        warnings.warn(
            f"About to compute diplotype matrix for {n_variants:,} variants "
            f"and {n_samples:,} samples; estimated memory {estimated_mb:.1f} MB.",
            UserWarning,
            stacklevel=2,
        )

        # Compute diplotype matrix: number of ref alleles per genotype call.
        gt = allel.GenotypeDaskArray(ds["call_genotype"].data)
        with self._dask_progress(desc="Compute diplotypes for LD pruning"):
            gn = gt.to_n_ref().compute()

        # Run LD pruning.
        with self._spinner(desc="LD pruning"):
            loc_unlinked = allel.locate_unlinked(
                gn, size=size, step=step, threshold=threshold
            )

        return dict(loc_unlinked=loc_unlinked)
