from typing import Optional
import os
import allel  # type: ignore
import numpy as np

from .snp_data import AnophelesSnpData
from . import base_params
from . import plink_params
from . import pca_params
from numpydoc_decorator import doc  # type: ignore


class VcfExporter(
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
            Write Anopheles biallelic SNP data to VCF file format.
        """,
        extended_summary="""
            This function writes biallelic SNPs to the VCF (Variant Call Format) file format.
            It enables subsetting to specific regions (`region`), selecting specific sample sets,
            or lists of samples, randomly downsampling sites, and specifying filters based on
            missing data and minimum minor allele count (see the docs for `biallelic_snp_calls`
            for more information). The `overwrite` parameter, set to true, will enable overwrite
            of data with the same SNP selection parameter values.
        """,
        returns="""
            Path to the output VCF file.
        """,
        notes="""
            This computation may take some time to run, depending on your computing
            environment. Unless the `overwrite` parameter is set to `True`, results will be
            returned from a previous computation, if available. The output follows the VCF 4.1
            specification and can be used directly with tools such as bcftools, GATK, and R
            packages like VariantAnnotation.
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

        # Define output file path
        vcf_file_path = (
            f"{output_dir}/{region}.{n_snps}.{min_minor_ac}"
            f".{max_missing_an}.{thin_offset}.vcf"
        )

        # Check to see if file exists and if overwrite is set to false, return existing file
        if os.path.exists(vcf_file_path):
            if not overwrite:
                return vcf_file_path

        # Get biallelic SNPs
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

        # Compute genotype array from dask
        with self._dask_progress("Computing genotype array"):
            gt_array = allel.GenotypeDaskArray(
                ds_snps["call_genotype"].data
            ).compute()

        # Extract variant info
        with self._spinner("Preparing VCF output data"):
            chrom = ds_snps["variant_contig"].values
            pos = ds_snps["variant_position"].values
            alleles = ds_snps["variant_allele"].values
            ref = alleles[:, 0]
            alt = alleles[:, 1]
            sample_ids = ds_snps["sample_id"].values

        # Build output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Write VCF file
        with self._spinner("Writing VCF file"):
            with open(vcf_file_path, "w") as vcf_out:
                # Write VCF header
                vcf_out.write("##fileformat=VCFv4.1\n")
                vcf_out.write(
                    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
                )
                # Write column header line
                sample_cols = "\t".join(sample_ids)
                vcf_out.write(
                    f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_cols}\n"
                )

                # Write one line per variant
                n_variants = len(pos)
                for i in range(n_variants):
                    gt_row = gt_array[i]  # shape: (n_samples, ploidy)
                    # Convert genotype calls to VCF GT string e.g. "0/1"
                    gt_strings = []
                    for sample_gt in gt_row:
                        if -1 in sample_gt:
                            gt_strings.append("./.")
                        else:
                            gt_strings.append("/".join(str(a) for a in sample_gt))
                    gt_col = "\t".join(gt_strings)
                    vcf_out.write(
                        f"{chrom[i]}\t{pos[i]}\t.\t{ref[i]}\t{alt[i]}"
                        f"\t.\tPASS\t.\tGT\t{gt_col}\n"
                    )

        return vcf_file_path