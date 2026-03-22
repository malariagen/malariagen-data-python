from typing import Optional

import numpy as np

import allel  # type: ignore
import xarray as xr
from numpydoc_decorator import doc  # type: ignore

from ..util import _check_types, _dask_compress_dataset
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
            Access biallelic SNP calls after LD pruning.
        """,
        extended_summary="""
            This function obtains biallelic SNP calls, then performs LD pruning
            using scikit-allel's `locate_unlinked` function. The resulting dataset
            can be used as input to ADMIXTURE workflows or exported to PLINK format.

            LD pruning is controlled by three parameters:

            - `ld_window_size`: number of SNPs in the sliding window used to
              compute pairwise r-squared.
            - `ld_window_step`: number of SNPs to advance the window each
              iteration.
            - `ld_threshold`: maximum r-squared value; SNP pairs above this
              are considered linked and one will be removed.

            Note that `n_snps` is required to control memory usage. Without
            pre-thinning, LD pruning could attempt to materialise millions of
            variants and run out of memory.
        """,
        returns="""
            A dataset of LD-pruned biallelic SNP calls with the same structure as
            the output of `biallelic_snp_calls`.
        """,
    )
    def biallelic_snp_calls_ld_pruned(
        self,
        region: base_params.regions,
        n_snps: base_params.n_snps,
        ld_window_size: ld_params.ld_window_size = ld_params.ld_window_size_default,
        ld_window_step: ld_params.ld_window_step = ld_params.ld_window_step_default,
        ld_threshold: ld_params.ld_threshold = ld_params.ld_threshold_default,
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
        random_seed: base_params.random_seed = 42,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
    ) -> xr.Dataset:
        # Check that either sample_query xor sample_indices are provided.
        base_params._validate_sample_selection_params(
            sample_query=sample_query, sample_indices=sample_indices
        )

        # Validate LD parameters.
        if ld_window_size <= 0:
            raise ValueError(f"ld_window_size must be > 0, got {ld_window_size}")
        if ld_window_step <= 0:
            raise ValueError(f"ld_window_step must be > 0, got {ld_window_step}")
        if not (0 < ld_threshold <= 1):
            raise ValueError(f"ld_threshold must be in (0, 1], got {ld_threshold}")

        # Obtain biallelic SNP calls with thinning applied first.
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
            inline_array=inline_array,
            chunks=chunks,
        )

        # Compute genotype reference counts.
        with self._dask_progress(desc="Computing genotype ref counts"):
            gt = ds_snps["call_genotype"].data
            gn = allel.GenotypeDaskArray(gt).to_n_ref(fill=-127).compute()

        # Perform LD pruning.
        with self._spinner(desc="LD pruning"):
            loc_unlinked = allel.locate_unlinked(
                gn,
                size=ld_window_size,
                step=ld_window_step,
                threshold=ld_threshold,
            )

        # Guard against empty result.
        if not np.any(loc_unlinked):
            raise ValueError(
                "LD pruning removed all variants. Consider using a less "
                "stringent ld_threshold or providing more variants via n_snps."
            )

        # Apply the pruning mask.
        ds_pruned = _dask_compress_dataset(
            ds_snps, indexer=loc_unlinked, dim="variants"
        )

        return ds_pruned
