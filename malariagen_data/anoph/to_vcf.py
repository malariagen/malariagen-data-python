from typing import Optional

import numpy as np
import os

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

        with self._spinner("Prepare output data"):
            genotypes = ds_snps["call_genotype"].values

            # Map integer contig indices to their string names (e.g., 0 -> '2L')
            chrom_indices = ds_snps["variant_contig"].values
            chrom = np.array(self.contigs)[chrom_indices]

            pos = ds_snps["variant_position"].values

            # Decode byte strings to unicode strings
            alleles = ds_snps["variant_allele"].values.astype("U")
            samples = ds_snps["sample_id"].values

            ref = alleles[:, 0]
            alt = alleles[:, 1]

            # Write VCF manually since allel.write_vcf does not support
            # sample/genotype data.
            with open(vcf_file_path, "w", encoding="utf-8") as f:
                # Write VCF header.
                print("##fileformat=VCFv4.1", file=f)
                print(
                    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
                    file=f,
                )

                # Write column header line with sample IDs.
                fixed_cols = [
                    "#CHROM",
                    "POS",
                    "ID",
                    "REF",
                    "ALT",
                    "QUAL",
                    "FILTER",
                    "INFO",
                    "FORMAT",
                ]
                header_fields = fixed_cols + [str(s) for s in samples]
                print("\t".join(header_fields), file=f)

                # Write data rows.
                for i in range(len(pos)):
                    # Format genotype calls for each sample.
                    gt_calls = []
                    for j in range(genotypes.shape[1]):
                        a0 = genotypes[i, j, 0]
                        a1 = genotypes[i, j, 1]
                        if a0 < 0 or a1 < 0:
                            gt_calls.append("./.")
                        else:
                            gt_calls.append(f"{a0}/{a1}")

                    row = [
                        str(chrom[i]),
                        str(pos[i]),
                        ".",
                        str(ref[i]),
                        str(alt[i]),
                        ".",
                        ".",
                        ".",
                        "GT",
                    ] + gt_calls
                    print("\t".join(row), file=f)

        return vcf_file_path
