from typing import Optional

import allel  # type: ignore
import numpy as np
import os
import bed_reader

from ..util import dask_compress_dataset
from .snp_data import AnophelesSnpData
from . import base_params
from . import plink_params
from . import pca_params
from numpydoc_decorator import doc  # type: ignore


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

    @doc(
        summary="""
            Write Anopheles biallelic SNP data to the Plink binary file format.
        """,
        extended_summary="""
            This function writes biallelic SNPs to the Plink binary file format. It enables
            subsetting to specific regions (`region`), selecting specific sample sets, or lists of
            samples, randomly downsampling sites, and specifying filters based on missing data and
            minimum minor allele count (see the docs for `biallelic_snp_calls` for more information).
            The `overwrite` parameter, set to true, will enable overwrite of data with the same
            SNP selection parameter values.
        """,
        returns="""
        Base path to files containing binary Plink output files. Append .bed,
        .bim or .fam to obtain paths for the binary genotype table file, variant
        information file and sample information file respectively.
        """,
        notes="""
            This computation may take some time to run, depending on your computing
            environment. Unless the `overwrite` parameter is set to `True`, results will be returned
            from a previous computation, if available.
        """,
    )
    def biallelic_snps_to_plink(
        self,
        output_dir: plink_params.output_dir,
        region: base_params.regions,
        n_snps: base_params.n_snps,
        overwrite: plink_params.overwrite = False,
        thin_offset: base_params.thin_offset = 0,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
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
    ):
        # Define output files
        plink_file_path = f"{output_dir}/{region}.{n_snps}.{min_minor_ac}.{max_missing_an}.{thin_offset}"

        bed_file_path = f"{plink_file_path}.bed"

        # Check to see if file exists and if overwrite is set to false, return existing file
        if os.path.exists(bed_file_path):
            if not overwrite:
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

        return plink_file_path
