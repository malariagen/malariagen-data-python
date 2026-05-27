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

# Supported FORMAT fields and their VCF header definitions.
_VALID_FIELDS = {"GT", "GQ", "AD", "MQ"}
_FORMAT_HEADERS = {
    "GT": '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    "GQ": '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">',
    "AD": '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele Depth">',
    "MQ": '##FORMAT=<ID=MQ,Number=1,Type=Integer,Description="Mapping Quality">',
}


class SnpVcfExporter(
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
        fields: vcf_params.vcf_fields = ("GT",),
    ) -> str:
        base_params._validate_sample_selection_params(
            sample_query=sample_query, sample_indices=sample_indices
        )

        # Validate fields parameter.
        fields = tuple(fields)
        unknown = set(fields) - _VALID_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown FORMAT fields: {unknown}. "
                f"Valid fields are: {sorted(_VALID_FIELDS)}"
            )
        if "GT" not in fields:
            raise ValueError("GT must be included in fields.")

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

        # Determine which extra fields to include.
        include_gq = "GQ" in fields
        include_ad = "AD" in fields
        include_mq = "MQ" in fields
        format_str = ":".join(fields)

        with opener(output_path, "wt") as f:
            # Write VCF header.
            f.write("##fileformat=VCFv4.3\n")
            f.write(f"##fileDate={date.today().strftime('%Y%m%d')}\n")
            f.write("##source=malariagen_data\n")
            for contig in contigs:
                f.write(f"##contig=<ID={contig}>\n")
            for field in fields:
                f.write(_FORMAT_HEADERS[field] + "\n")
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

            # Extract dask arrays.
            gt_data = ds["call_genotype"].data
            pos_data = ds["variant_position"].data
            contig_data = ds["variant_contig"].data
            allele_data = ds["variant_allele"].data

            # Optional field arrays — may not exist in all datasets.
            gq_data = None
            ad_data = None
            mq_data = None
            if include_gq:
                try:
                    gq_data = ds["call_GQ"].data
                except KeyError:
                    pass
            if include_ad:
                try:
                    ad_data = ds["call_AD"].data
                except KeyError:
                    pass
            if include_mq:
                try:
                    mq_data = ds["call_MQ"].data
                except KeyError:
                    pass

            chunk_sizes = gt_data.chunks[0]
            offsets = np.cumsum((0,) + chunk_sizes)

            # Write records in chunks.
            with self._spinner(f"Write VCF ({ds.sizes['variants']} variants)"):
                for ci in range(len(chunk_sizes)):
                    start = offsets[ci]
                    stop = offsets[ci + 1]
                    gt_chunk = gt_data[start:stop].compute()
                    pos_chunk = pos_data[start:stop].compute()
                    contig_chunk = contig_data[start:stop].compute()
                    allele_chunk = allele_data[start:stop].compute()

                    # Compute optional field chunks, handling missing data.
                    gq_chunk = None
                    ad_chunk = None
                    mq_chunk = None
                    if gq_data is not None:
                        try:
                            gq_chunk = gq_data[start:stop].compute()
                        except (FileNotFoundError, KeyError):
                            pass
                    if ad_data is not None:
                        try:
                            ad_chunk = ad_data[start:stop].compute()
                        except (FileNotFoundError, KeyError):
                            pass
                    if mq_data is not None:
                        try:
                            mq_chunk = mq_data[start:stop].compute()
                        except (FileNotFoundError, KeyError):
                            pass

                    n_samples = gt_chunk.shape[1]

                    # OPTIMIZATION: Vectorize GT field formatting across entire chunk.
                    # Instead of formatting each sample's GT field in a nested Python loop
                    # (which results in billions of string operations for large datasets),
                    # use NumPy's vectorized string operations on the entire chunk at once.
                    # This provides ~3x speedup while maintaining exact output compatibility.
                    # See issue #1280 for performance analysis.
                    gt_chunk_2d = gt_chunk.reshape(
                        gt_chunk.shape[0], gt_chunk.shape[1], 2
                    )
                    a0 = gt_chunk_2d[:, :, 0]  # (n_variants, n_samples)
                    a1 = gt_chunk_2d[:, :, 1]  # (n_variants, n_samples)
                    missing = (a0 < 0) | (a1 < 0)

                    # Build formatted GT strings using NumPy vectorization
                    gt_formatted = np.empty(
                        (gt_chunk.shape[0], n_samples), dtype=object
                    )
                    gt_formatted[missing] = "./."
                    present_idx = ~missing
                    if np.any(present_idx):
                        a0_str = a0[present_idx].astype(str)
                        a1_str = a1[present_idx].astype(str)
                        gt_formatted[present_idx] = np.char.add(
                            np.char.add(a0_str, "/"), a1_str
                        )

                    # Pre-allocate line buffer for better I/O
                    lines_to_write = []

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

                        # Build fixed VCF columns once per variant
                        fixed_cols = (
                            f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\t{format_str}\t"
                        )

                        sample_fields = np.empty(n_samples, dtype=object)

                        # Use pre-formatted GT strings and add other fields
                        for k in range(n_samples):
                            parts = [gt_formatted[j, k]]

                            # GQ.
                            if include_gq:
                                if gq_chunk is not None:
                                    v = gq_chunk[j, k]
                                    parts.append("." if v < 0 else str(v))
                                else:
                                    parts.append(".")
                            # AD.
                            if include_ad:
                                if ad_chunk is not None:
                                    ad_vals = ad_chunk[j, k]
                                    parts.append(
                                        ",".join(
                                            "." if x < 0 else str(x) for x in ad_vals
                                        )
                                    )
                                else:
                                    parts.append(".")
                            # MQ.
                            if include_mq:
                                if mq_chunk is not None:
                                    v = mq_chunk[j, k]
                                    parts.append("." if v < 0 else str(v))
                                else:
                                    parts.append(".")
                            sample_fields[k] = ":".join(parts)

                        # Build and buffer the line
                        line = fixed_cols + "\t".join(sample_fields) + "\n"
                        lines_to_write.append(line)

                    # Write buffered lines in one go per chunk
                    f.write("".join(lines_to_write))

        return output_path
