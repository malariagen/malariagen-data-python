import warnings
from typing import Optional

import pandas as pd
from numpydoc_decorator import doc  # type: ignore

from ..util import CacheMiss, _check_types
from . import base_params, pca_params, relatedness_params
from .pca import AnophelesPca


class AnophelesRelatednessAnalysis(
    AnophelesPca,
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
            Compute PC-Relate kinship coefficient matrix using sgkit's implementation.
        """,
        extended_summary="""
            This method computes the kinship coefficient matrix. The kinship coefficient for
            a pair of individuals ``i`` and ``j`` is commonly defined to be the probability that
            a random allele selected from ``i`` and a random allele selected from ``j`` at
            a locus are IBD.
        """,
        returns="df_kinship",
        notes="""
            This computation may take some time to run, depending on your computing
            environment. Results of this computation will be cached and re-used if
            the `results_cache` parameter was set when instantiating the API client.
        """,
    )
    def pc_relate(
        self,
        region: base_params.regions,
        n_snps: base_params.n_snps,
        n_components: pca_params.n_components = pca_params.n_components_default,
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
        imputation_method: pca_params.imputation_method = pca_params.imputation_method_default,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
        exclude_samples: Optional[base_params.samples] = None,
        fit_exclude_samples: Optional[base_params.samples] = None,
        random_seed: base_params.random_seed = 42,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
        maf: relatedness_params.maf = relatedness_params.maf_default,
    ) -> pd.DataFrame:
        # Change this name if you ever change the behaviour of this function, to
        # invalidate any previously cached data.
        name = "pc_relate_v1"

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
            imputation_method=imputation_method,
            n_components=n_components,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            exclude_samples=exclude_samples,
            fit_exclude_samples=fit_exclude_samples,
            random_seed=random_seed,
            maf=maf,
        )

        # Try to retrieve results from the cache.
        try:
            results = self.results_cache_get(name=name, params=params)

        except CacheMiss:
            results = self._pc_relate(
                chunks=chunks, inline_array=inline_array, **params
            )
            self.results_cache_set(name=name, params=params, results=results)

        df_kinship = pd.DataFrame(
            results["kinship_matrix"],
            index=results["samples"],
            columns=results["samples"],
        )

        # Name the index to match other datasets
        df_kinship.index.name = "sample_id"
        df_kinship.columns.name = "partner_sample_id"

        return df_kinship

    def _pc_relate(
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
        imputation_method="most_common",
        n_components,
        cohort_size,
        min_cohort_size,
        max_cohort_size,
        exclude_samples,
        fit_exclude_samples,
        random_seed,
        maf,
        chunks,
        inline_array,
    ):
        import numpy as np

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # sgkit has warnings we don't need
            import sgkit

        # First, run the internal `_pca` to obtain PCA coords mapping.
        # This will return coords internally aligned with `samples`.
        pca_results = self._pca(
            region=region,
            n_snps=n_snps,
            thin_offset=thin_offset,
            sample_sets=sample_sets,
            sample_indices=sample_indices,
            site_mask=site_mask,
            site_class=site_class,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            imputation_method=imputation_method,
            n_components=n_components,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
            exclude_samples=exclude_samples,
            fit_exclude_samples=fit_exclude_samples,
            random_seed=random_seed,
            chunks=chunks,
            inline_array=inline_array,
        )

        pca_coords = pca_results["coords"]
        samples = pca_results["samples"].astype("U")

        # Now get the raw xarray genotypes to match the subsets
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

        ds_samples = ds["sample_id"].values.astype("U")

        # We need to filter `ds` so it matches the samples (since PCA excluded `exclude_samples`)
        if len(samples) < len(ds_samples):
            loc_keep = np.isin(ds_samples, samples)
            ds = ds.isel(samples=loc_keep)

        # Verify alignment
        if ds.sizes["samples"] != len(samples):
            raise AssertionError("Samples from PCA do not match subsetted dataset!")

        # Add `call_genotype_mask` (shape is variants x samples x alleles/ploidy)
        call_genotype = ds["call_genotype"]
        call_genotype_mask = call_genotype < 0
        ds["call_genotype_mask"] = call_genotype_mask

        # Add `sample_pca_projection` for sgkit.
        ds["sample_pca_projection"] = (["samples", "components"], pca_coords)

        # Run PC-Relate
        with self._spinner(desc="Compute PC-Relate Kinship"):
            ds_rel = sgkit.pc_relate(ds, maf=maf)
            kinship_matrix = ds_rel["pc_relate_phi"].values

        results = dict(
            samples=samples,
            kinship_matrix=kinship_matrix,
        )
        return results
