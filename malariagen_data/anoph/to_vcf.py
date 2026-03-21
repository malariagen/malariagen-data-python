from typing import Optional
import os

from .snp_data import AnophelesSnpData
from . import base_params
from . import plink_params
from . import pca_params
from numpydoc_decorator import doc  # type: ignore
import dask.array as da


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
            It enables subsetting to specific regions (region), selecting specific sample sets,
            or lists of samples, randomly downsampling sites, and specifying filters based on
            missing data and minimum minor allele count (see the docs for biallelic_snp_calls
            for more information). The overwrite parameter, set to true, will enable overwrite
            of data with the same SNP selection parameter values.
        """,
        returns="""
            Path to the output VCF file.
        """,
        notes="""
            This computation may take some time to run, depending on your computing
            environment. Unless the overwrite parameter is set to True, results will be
            returned from a previous computation, if available. The output follows the VCF 4.1
            specification and can be used directly with tools such as bcftools, GATK, and R
            packages like VariantAnnotation.

            Genotype data is written chunk-by-chunk directly from Dask arrays, so the full
            genotype matrix is never loaded into memory at once. This is important for
            large datasets where VCF files can be very large.
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

        # Return existing file if overwrite is False
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

        # Extract variant and sample metadata (small arrays, safe to load fully)
        with self._spinner("Preparing VCF metadata"):
            # variant_contig contains integer indices, convert to contig name strings
            contig_indices = ds_snps["variant_contig"].values
            chrom = [self.contigs[i] for i in contig_indices]
            pos = ds_snps["variant_position"].values
            alleles = ds_snps["variant_allele"].values
            ref = alleles[:, 0]
            alt = alleles[:, 1]
            sample_ids = ds_snps["sample_id"].values

        # Build output directory if it does not exist
        os.makedirs(output_dir, exist_ok=True)

        # Write VCF file using chunked iteration over Dask arrays.
        # This avoids loading the full genotype matrix into memory at once,
        # which is critical given the potential size of VCF files.
        gt_dask = ds_snps["call_genotype"].data  # shape: (variants, samples, ploidy)
        gt_chunks = da.rechunk(gt_dask, chunks=(1000, -1, -1))

        with self._spinner("Writing VCF file"):
            with open(vcf_file_path, "w", encoding="utf-8") as f:
                # Write VCF header
                f.write("##fileformat=VCFv4.1\n")
                f.write('##FILTER=<ID=PASS,Description="All filters passed">\n')
                f.write(
                    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
                )
                f.write(
                    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
                    + "\t".join(str(s) for s in sample_ids)
                    + "\n"
                )

                # Iterate chunk by chunk - 1000 variants at a time
                chunk_start = 0
                for chunk in gt_chunks.blocks:
                    gt_block = chunk.compute()
                    chunk_size = gt_block.shape[0]
                    for i in range(chunk_size):
                        v = chunk_start + i
                        gt_strings = []
                        for s in range(gt_block.shape[1]):
                            a0 = gt_block[i, s, 0]
                            a1 = gt_block[i, s, 1]
                            if a0 < 0 or a1 < 0:
                                gt_strings.append("./.")
                            else:
                                gt_strings.append(f"{a0}/{a1}")
                        ref_val = ref[v]
                        alt_val = alt[v]
                        if isinstance(ref_val, bytes):
                            ref_val = ref_val.decode()
                        if isinstance(alt_val, bytes):
                            alt_val = alt_val.decode()
                        f.write(
                            f"{chrom[v]}\t{pos[v]}\t.\t{ref_val}\t{alt_val}\t.\tPASS\t.\tGT\t"
                            + "\t".join(gt_strings)
                            + "\n"
                        )
                    chunk_start += chunk_size

        return vcf_file_path
