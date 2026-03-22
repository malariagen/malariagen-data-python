"""VCF export for Anopheles SNP call data.

This module provides a VcfConverter mixin that adds the ability to export
SNP call datasets to VCF (Variant Call Format), enabling interoperability
with tools such as bcftools, GATK, and R genomics pipelines.
"""

import os
from typing import Optional

from numpydoc_decorator import doc  # type: ignore

from .snp_data import AnophelesSnpData
from . import base_params
from . import vcf_params


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
            Export Anopheles SNP call data to Variant Call Format (VCF).
        """,
        extended_summary="""
            This function writes SNP calls to a VCF file. It supports subsetting
            to specific regions, selecting specific sample sets or individual
            samples, and applying site filters. Data is streamed in chunks to
            keep memory usage low when working with large datasets.
        """,
        returns="""
        Path to the output VCF file.
        """,
        notes="""
            This computation may take some time to run, depending on your
            computing environment and the size of the selected region. Unless
            the ``overwrite`` parameter is set to ``True``, results will be
            returned from a previous computation if the output file already
            exists.
        """,
    )
    def snp_calls_to_vcf(
        self,
        output_dir: vcf_params.output_dir,
        region: base_params.regions,
        overwrite: vcf_params.overwrite = False,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        sample_indices: Optional[base_params.sample_indices] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
        out: Optional[vcf_params.out] = None,
    ) -> str:
        # Check that either sample_query xor sample_indices are provided.
        base_params._validate_sample_selection_params(
            sample_query=sample_query, sample_indices=sample_indices
        )

        # Build output file path.
        if out is not None:
            vcf_file_path = f"{output_dir}/{out}.vcf"
        else:
            vcf_file_path = f"{output_dir}/{region}.vcf"

        # Check to see if file exists and if overwrite is set to false,
        # return existing file.
        if os.path.exists(vcf_file_path):
            if not overwrite:
                return vcf_file_path

        # Ensure output directory exists.
        os.makedirs(output_dir, exist_ok=True)

        # Load SNP call data.
        ds = self.snp_calls(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            sample_indices=sample_indices,
            site_mask=site_mask,
            inline_array=inline_array,
            chunks=chunks,
        )

        # Extract sample IDs.
        sample_ids = ds["sample_id"].values
        # Decode bytes to strings if needed.
        if sample_ids.dtype.kind == "S":
            sample_ids = sample_ids.astype("U")

        # Extract variant data lazily (dask arrays).
        variant_position = ds["variant_position"]
        variant_contig = ds["variant_contig"]
        variant_allele = ds["variant_allele"]
        call_genotype = ds["call_genotype"]

        n_variants = ds.sizes["variants"]

        # Determine chunk size for streaming writes.
        chunk_size = 10_000

        with self._spinner("Writing VCF"):
            with open(vcf_file_path, "w") as vcf_file:
                # ---- Write VCF header ----
                vcf_file.write("##fileformat=VCFv4.3\n")
                vcf_file.write(
                    "##FORMAT=<ID=GT,Number=1,Type=String," 'Description="Genotype">\n'
                )
                # Column header line.
                header_cols = [
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
                header_cols.extend(str(s) for s in sample_ids)
                vcf_file.write("\t".join(header_cols) + "\n")

                # ---- Write variant records in chunks ----
                for chunk_start in range(0, n_variants, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, n_variants)
                    chunk_slice = slice(chunk_start, chunk_end)

                    # Compute this chunk (materialise from dask).
                    pos_chunk = variant_position[chunk_slice].values
                    contig_chunk = variant_contig[chunk_slice].values
                    allele_chunk = variant_allele[chunk_slice].values
                    gt_chunk = call_genotype[chunk_slice].values

                    # Decode bytes to strings if needed.
                    if contig_chunk.dtype.kind == "S":
                        contig_chunk = contig_chunk.astype("U")
                    if allele_chunk.dtype.kind == "S":
                        allele_chunk = allele_chunk.astype("U")

                    n_chunk = pos_chunk.shape[0]

                    for i in range(n_chunk):
                        chrom = str(contig_chunk[i])
                        pos = str(pos_chunk[i])

                        # REF is the first allele, ALT are the remaining
                        # non-empty alleles.
                        alleles = allele_chunk[i]
                        ref = str(alleles[0])
                        alt_alleles = [
                            str(a)
                            for a in alleles[1:]
                            if str(a) != "" and str(a) != "."
                        ]
                        alt = ",".join(alt_alleles) if alt_alleles else "."

                        # Format genotypes as VCF GT strings.
                        gt_row = gt_chunk[i]  # shape: (n_samples, ploidy)
                        gt_strings = []
                        for j in range(gt_row.shape[0]):
                            a0 = gt_row[j, 0]
                            a1 = gt_row[j, 1]
                            if a0 < 0 or a1 < 0:
                                gt_strings.append("./.")
                            else:
                                gt_strings.append(f"{a0}/{a1}")

                        # Write the VCF record line.
                        fields = [
                            chrom,
                            pos,
                            ".",  # ID
                            ref,
                            alt,
                            ".",  # QUAL
                            ".",  # FILTER
                            ".",  # INFO
                            "GT",  # FORMAT
                        ]
                        fields.extend(gt_strings)
                        vcf_file.write("\t".join(fields) + "\n")

        return vcf_file_path
