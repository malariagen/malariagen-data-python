from typing import Optional

import allel  # type: ignore
import numpy as np
import os
import bed_reader

from util import dask_compress_dataset
from .snp_data import AnophelesSnpData
from . import base_params


class PlinkConverter(
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

    def _create_plink_outfile(
        self,
        *,
        results_dir,
        region,
        n_snps,
        min_minor_ac,
        thin_offset,
        max_missing_an,
    ):
        return f"{results_dir}/{region}.{n_snps}.{min_minor_ac}.{thin_offset}.{max_missing_an}"

    def _biallelic_snps_to_plink(
        self,
        *,
        results_dir,
        region,
        n_snps,
        thin_offset,
        sample_sets,
        sample_query,
        sample_indices,
        site_mask,
        min_minor_ac,
        max_missing_an,
        random_seed,
        inline_array,
        chunks,
    ):
        # Define output file
        plink_file_path = self._create_plink_outfile(
            results_dir=results_dir,
            region=region,
            n_snps=n_snps,
            thin_offset=thin_offset,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
        )

        bed_file_path = f"{plink_file_path}.bed"
        if os.path.exists(bed_file_path):
            return plink_file_path

        # Get snps
        ds_snps = self.biallelic_snp_calls(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
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

        # Set up dataset with required vars for plink conversion

        # Compute gt ref counts
        with self._dask_progress("Computing genotype ref counts"):
            gt_asc = ds_snps["call_genotype"].data  # dask array
            gn_ref = allel.GenotypeDaskArray(gt_asc).to_n_ref(fill=-127)
            gn_ref = gn_ref.compute()

        # Ensure genotypes vary
        loc_var = np.any(gn_ref != gn_ref[:, 0, np.newaxis], axis=1)

        # Load final data
        ds_snps_final = dask_compress_dataset(ds_snps, loc_var, dim="variants")

        # Init vars for input to bed reader
        gn_ref_final = gn_ref[loc_var]
        val = gn_ref_final.T
        with self._spinner("Prepare output data"):
            alleles = ds_snps_final["variant_allele"].values
            properties = {
                "iid": ds_snps_final["sample_id"].values,
                "chromosome": ds_snps_final["variant_contig"].values,
                "bp_position": ds_snps_final["variant_position"].values,
                "allele_1": alleles[:, 0],
                "allele_2": alleles[:, 1],
            }

        bed_reader.to_bed(
            filepath=bed_file_path,
            val=val,
            properties=properties,
            count_A1=True,
        )

        print(f"PLINK files written to to: {plink_file_path}")

        return plink_file_path

    def biallelic_snps_to_plink(
        self,
        results_dir,
        region: base_params.regions,
        n_snps: base_params.n_snps,
        thin_offset: base_params.thin_offset = 0,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_indices: Optional[base_params.sample_indices] = None,
        site_mask: Optional[base_params.site_mask] = None,
        min_minor_ac: Optional[base_params.min_minor_ac] = 0,
        max_missing_an: Optional[base_params.max_missing_an] = 0,
        random_seed: base_params.random_seed = 42,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
    ):
        params = dict(
            results_dir=results_dir,
            region=region,
            n_snps=n_snps,
            thin_offset=thin_offset,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_indices=sample_indices,
            site_mask=site_mask,
            min_minor_ac=min_minor_ac,
            max_missing_an=max_missing_an,
            random_seed=random_seed,
        )

        filepath = self._biallelic_snps_to_plink(
            inline_array=inline_array, chunks=chunks, **params
        )
        return filepath
