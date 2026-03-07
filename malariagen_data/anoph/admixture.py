from typing import Optional

import allel  # type: ignore
import numpy as np
import os

import bed_reader

from numpydoc_decorator import doc  # type: ignore

from ..util import _check_types, _dask_compress_dataset
from . import base_params, pca_params, plink_params, ld_pruning_params, admixture_params
from .ld_pruning import AnophelesLdPruning


class AnophelesAdmixture(
    AnophelesLdPruning,
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
            Prepare LD-pruned, pre-filtered data for ADMIXTURE analysis.
        """,
        extended_summary="""
            This function performs LD pruning, optional pre-filtering
            (random downsampling of SNPs), and writes the result
            to PLINK binary format (.bed/.bim/.fam) compatible with
            the ADMIXTURE software. The output files can be used
            directly as input to the ADMIXTURE command-line tool.

            The PLINK output follows the same conventions as
            ``biallelic_snps_to_plink()``.
        """,
        returns="""
            A dict containing the base path to the PLINK output files
            (append .bed, .bim, or .fam to get individual file paths),
            along with a summary of filtering statistics.
        """,
        notes="""
            This computation may take some time to run, depending on your
            computing environment. Unless the ``overwrite`` parameter is set
            to ``True``, results will be returned from a previous computation,
            if available.
        """,
    )
    def prepare_admixture(
        self,
        output_dir: plink_params.output_dir,
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
        max_snps: Optional[admixture_params.max_snps] = None,
        overwrite: plink_params.overwrite = False,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
    ) -> dict:
        # Check that either sample_query xor sample_indices are provided.
        base_params._validate_sample_selection_params(
            sample_query=sample_query, sample_indices=sample_indices
        )

        # Create output directory if it doesn't exist.
        os.makedirs(output_dir, exist_ok=True)

        # Define output file path.
        plink_file_path = f"{output_dir}/admixture.{region}.r2_{r2_threshold}"
        bed_file_path = f"{plink_file_path}.bed"

        # Check for existing output.
        if os.path.exists(bed_file_path) and not overwrite:
            return dict(
                plink_path=plink_file_path,
                from_cache=True,
            )

        # Step 1: LD prune the data (returns xr.Dataset).
        ds_pruned = self.ld_prune(
            region=region,
            n_snps=n_snps,
            thin_offset=thin_offset,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            sample_indices=sample_indices,
            site_mask=site_mask,
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
            inline_array=inline_array,
            chunks=chunks,
        )

        n_snps_after_ld = ds_pruned.sizes["variants"]

        # Step 2: Optional random SNP downsampling.
        if max_snps is not None and n_snps_after_ld > max_snps:
            rng = np.random.RandomState(random_seed)
            snp_indices = np.sort(
                rng.choice(n_snps_after_ld, size=max_snps, replace=False)
            )
            loc_downsample = np.zeros(n_snps_after_ld, dtype=bool)
            loc_downsample[snp_indices] = True
            ds_pruned = _dask_compress_dataset(
                ds_pruned, indexer=loc_downsample, dim="variants"
            )

        # Step 3: Write PLINK binary files.
        # Follow the same conventions as biallelic_snps_to_plink() in to_plink.py.

        # Compute genotype ref counts.
        with self._dask_progress("Computing genotype ref counts"):
            gt_asc = ds_pruned["call_genotype"].data  # dask array
            gn_ref = allel.GenotypeDaskArray(gt_asc).to_n_ref(fill=-127)
            gn_ref = gn_ref.compute()

        # Ensure genotypes vary.
        loc_var = np.any(gn_ref != gn_ref[:, 0, np.newaxis], axis=1)

        # Load final data.
        ds_final = _dask_compress_dataset(ds_pruned, loc_var, dim="variants")

        # Prepare data for bed_reader.
        gn_ref_final = gn_ref[loc_var]
        val = gn_ref_final.T
        with self._spinner("Writing PLINK files for ADMIXTURE"):
            alleles = ds_final["variant_allele"].values
            properties = {
                "iid": ds_final["sample_id"].values,
                "chromosome": ds_final["variant_contig"].values,
                "bp_position": ds_final["variant_position"].values,
                "allele_1": alleles[:, 0],
                "allele_2": alleles[:, 1],
            }

            bed_reader.to_bed(
                filepath=bed_file_path,
                val=val,
                properties=properties,
                count_A1=True,
            )

        results = dict(
            plink_path=plink_file_path,
            from_cache=False,
            n_samples=ds_final.sizes["samples"],
            n_snps_after_ld=n_snps_after_ld,
            n_snps_final=ds_final.sizes["variants"],
        )

        return results
