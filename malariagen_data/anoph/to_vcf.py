from typing import Optional

import allel  # type: ignore
import numpy as np
import os

from ..util import _dask_compress_dataset
from .snp_data import AnophelesSnpData
from . import base_params
from . import plink_params
from . import pca_params
from numpydoc_decorator import doc  # type: ignore


class VcfConverter(
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
            Write Anopheles biallelic SNP data to the VCF text file format.
        """,
        extended_summary="""
            This function writes biallelic SNPs to the Variant Call Format (VCF). It enables
            subsetting to specific regions (`region`), selecting specific sample sets, or lists of
            samples, randomly downsampling sites, and specifying filters based on missing data and
            minimum minor allele count (see the docs for `biallelic_snp_calls` for more information).
            The `overwrite` parameter, set to true, will enable overwrite of data with the same
            SNP selection parameter values.
        """,
        returns="""
        Path to the generated .vcf file.
        """,
        notes="""
            This computation may take some time to run, depending on your computing
            environment. Unless the `overwrite` parameter is set to `True`, results will be returned
            from a previous computation, if available.
        """,
    )
    def biallelic_snps_to_vcf(
        self,
        output_dir: plink_params.output_dir,
        region: base_params.regions,
        n_snps: base_params.n_snps,
        overwrite: plink_params.overwrite = False,
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
    ):
        # Check that either sample_query xor sample_indices are provided.
        base_params._validate_sample_selection_params(
            sample_query=sample_query, sample_indices=sample_indices
        )

        # Define output files
        vcf_file_path = f"{output_dir}/{region}.{n_snps}.{min_minor_ac}.{max_missing_an}.{thin_offset}.vcf"

        # Check to see if file exists and if overwrite is set to false, return existing file
        if os.path.exists(vcf_file_path):
            if not overwrite:
                return vcf_file_path

        # Get snps
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

        # Compute gt ref counts
        with self._dask_progress("Computing genotype ref counts"):
            gt_asc = ds_snps["call_genotype"].data  # dask array
            gn_ref = allel.GenotypeDaskArray(gt_asc).to_n_ref(fill=-127)
            gn_ref = gn_ref.compute()

        # Ensure genotypes vary
        loc_var = np.any(gn_ref != gn_ref[:, 0, np.newaxis], axis=1)

        # Load final data
        ds_snps_final = _dask_compress_dataset(ds_snps, loc_var, dim="variants")

        with self._spinner("Prepare output data"):
            genotypes = ds_snps_final["call_genotype"].values
            chrom = ds_snps_final["variant_contig"].values
            pos = ds_snps_final["variant_position"].values
            alleles = ds_snps_final["variant_allele"].values
            samples = ds_snps_final["sample_id"].values

            ref = alleles[:, 0]
            alt = [
                a[1:] for a in alleles
            ]  # Get the alternate alleles as a list of arrays

            # We format everything into a VCF using scikit-allel's optimized exporter
            with open(vcf_file_path, "w", encoding="utf-8") as f:
                allel.write_vcf(
                    f,
                    chrom=chrom,
                    pos=pos,
                    ref=ref,
                    alt=alt,
                    samples=samples,
                    calldata_2d={"GT": genotypes},
                )

        return vcf_file_path
