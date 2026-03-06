from typing import Optional

import allel  # type: ignore
import numpy as np

from numpydoc_decorator import doc  # type: ignore

from ..util import _check_types
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
            A numpy array of genotype values (n_snps, n_samples) after
            LD pruning, along with the pruned variant positions and
            sample IDs.
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
    ) -> dict:
        # Check that either sample_query xor sample_indices are provided.
        base_params._validate_sample_selection_params(
            sample_query=sample_query, sample_indices=sample_indices
        )

        # Load biallelic SNP calls.
        ds_snps = self.biallelic_snp_calls(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
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

        # Compute diplotype array (alt allele counts per individual).
        with self._dask_progress("Computing diplotypes for LD pruning"):
            gt = allel.GenotypeDaskArray(ds_snps["call_genotype"].data)
            gn = gt.to_n_alt(fill=-1).compute()

        # Remove sites with no variation.
        loc_var = np.any(gn != gn[:, 0, np.newaxis], axis=1)
        gn_var = gn[loc_var]

        # Iterative LD pruning using Rogers-Huff r².
        n_before = gn_var.shape[0]
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

        n_pruned = np.sum(loc_unlinked)

        # Map loc_unlinked back to full dataset (accounting for loc_var).
        var_indices = np.where(loc_var)[0]
        pruned_indices = var_indices[loc_unlinked]
        loc_pruned_full = np.zeros(ds_snps.sizes["variants"], dtype=bool)
        loc_pruned_full[pruned_indices] = True

        # Extract pruned data.
        sample_ids = ds_snps["sample_id"].values
        variant_positions = ds_snps["variant_position"].values[loc_pruned_full]
        variant_contigs = ds_snps["variant_contig"].values[loc_pruned_full]
        gn_pruned = gn[loc_var][loc_unlinked]

        results = dict(
            gn=gn_pruned,
            samples=sample_ids,
            variant_position=variant_positions,
            variant_contig=variant_contigs,
            n_snps_before=n_before,
            n_snps_after=int(n_pruned),
        )

        return results
