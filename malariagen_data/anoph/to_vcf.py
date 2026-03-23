import gzip
import os
from datetime import date
from typing import Optional

import numpy as np
from numpydoc_decorator import doc  # type: ignore

from .snp_data import AnophelesSnpData
from . import base_params
from . import plink_params
from . import vcf_params


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
            Export SNP calls to Variant Call Format (VCF).
        """,
        extended_summary="""
            This function writes SNP calls to a VCF file. Data is written
            in chunks to avoid loading the entire genotype matrix into
            memory. Supports optional gzip compression when the output
            path ends with `.gz`.
        """,
        returns="""
        Path to the VCF output file.
        """,
    )
    def snp_calls_to_vcf(
        self,
        output_path: vcf_params.vcf_output_path,
        region: base_params.regions,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        sample_indices: Optional[base_params.sample_indices] = None,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
        overwrite: plink_params.overwrite = False,
    ) -> str:
        base_params._validate_sample_selection_params(
            sample_query=sample_query, sample_indices=sample_indices
        )

        if os.path.exists(output_path) and not overwrite:
            return output_path

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

        sample_ids = ds["sample_id"].values
        contigs = ds.attrs.get("contigs", self.contigs)
        compress = output_path.endswith(".gz")
        opener = gzip.open if compress else open

        with opener(output_path, "wt") as f:
            # Write VCF header.
            f.write("##fileformat=VCFv4.3\n")
            f.write(f"##fileDate={date.today().strftime('%Y%m%d')}\n")
            f.write("##source=malariagen_data\n")
            for contig in contigs:
                f.write(f"##contig=<ID={contig}>\n")
            f.write('##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n')
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
            f.write("\t".join(header_cols + list(sample_ids)) + "\n")

            # Write records in chunks.
            gt_data = ds["call_genotype"].data
            pos_data = ds["variant_position"].data
            contig_data = ds["variant_contig"].data
            allele_data = ds["variant_allele"].data

            chunk_sizes = gt_data.chunks[0]
            offsets = np.cumsum((0,) + chunk_sizes)

            with self._spinner(f"Write VCF ({ds.sizes['variants']} variants)"):
                for ci in range(len(chunk_sizes)):
                    start = offsets[ci]
                    stop = offsets[ci + 1]
                    gt_chunk = gt_data[start:stop].compute()
                    pos_chunk = pos_data[start:stop].compute()
                    contig_chunk = contig_data[start:stop].compute()
                    allele_chunk = allele_data[start:stop].compute()

                    for j in range(gt_chunk.shape[0]):
                        chrom = contigs[contig_chunk[j]]
                        pos = str(pos_chunk[j])
                        alleles = allele_chunk[j]
                        ref = (
                            alleles[0].decode()
                            if hasattr(alleles[0], "decode")
                            else str(alleles[0])
                        )
                        alt_alleles = []
                        for a in alleles[1:]:
                            s = a.decode() if hasattr(a, "decode") else str(a)
                            if s:
                                alt_alleles.append(s)
                        alt = ",".join(alt_alleles) if alt_alleles else "."

                        gt_row = gt_chunk[j]
                        sample_fields = np.empty(gt_row.shape[0], dtype=object)
                        for k in range(gt_row.shape[0]):
                            a0 = gt_row[k, 0]
                            a1 = gt_row[k, 1]
                            if a0 < 0 or a1 < 0:
                                sample_fields[k] = "./."
                            else:
                                sample_fields[k] = f"{a0}/{a1}"

                        line = f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\tGT\t"
                        line += "\t".join(sample_fields)
                        f.write(line + "\n")

        return output_path
