import gzip
import os
import re
from datetime import date
from typing import Optional
import dask
from dask.array.core import Array
from xarray.core.dataset import Dataset
import numpy as np
from numpydoc_decorator import doc  # type: ignore
from dataclasses import dataclass
import math

from .snp_data import AnophelesSnpData
from . import base_params
from . import plink_params
from . import vcf_params


@dataclass
class VariantChunkData:
    gq_chunk: Array
    ad_chunk: Array
    mq_chunk: Array
    gt_chunk: Array
    pos_chunk: Array
    contig_chunk: Array
    allele_chunk: Array


# Supported FORMAT fields, the fixed order in which their values are
# always written per sample (regardless of what order the caller's
# `fields` argument happens to iterate in — e.g. a set has no
# guaranteed order at all), and their VCF header definitions.
_VALID_FIELDS = {"GT", "GQ", "AD", "MQ"}
_FIELD_ORDER = ("GT", "GQ", "AD", "MQ")
_FORMAT_HEADERS = {
    "GT": '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    "GQ": '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">',
    "AD": '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele Depth">',
    # N.B., "MQ" is a reserved FORMAT key in the VCF specification,
    # fixed there as Integer, Number=1 — even though the underlying
    # data is a float (RMS mapping quality), so values are rounded to
    # the nearest integer when written below to stay spec-compliant
    # under the reserved key's declared type.
    "MQ": '##FORMAT=<ID=MQ,Number=1,Type=Integer,Description="Mapping Quality">',
}

# snp_calls_to_vcf() only supports exporting a single sample at a time
# (its current use case is generating a per-sample VCF for IGV), so
# sample_query must be of the form "sample_id == '<sample_id>'".
_SINGLE_SAMPLE_QUERY_PATTERN = re.compile(
    r"""^\s*sample_id\s*==\s*(['"])[^'"]+\1\s*$"""
)


def _validate_single_sample_selection(
    *,
    sample_sets: base_params.sample_sets,
    sample_query: base_params.sample_query,
) -> None:
    """
    Validate that `sample_sets` and `sample_query` are provided and
    together specify a single sample, which is the only usage that
    `snp_calls_to_vcf()` currently supports. A VCF file is only
    meaningful with respect to an explicit sample basis for its calls,
    so both parameters are required; and because the only current use
    case is generating a per-sample VCF for IGV, `sample_query` is
    required to select exactly one sample.
    """
    if not sample_sets:
        raise ValueError(
            "sample_sets must be provided and non-empty: a VCF must be "
            "generated against an explicit set of samples."
        )
    if not sample_query or not sample_query.strip():
        raise ValueError(
            "sample_query must be provided and non-empty, and must "
            "select a single sample (see snp_calls_to_vcf() docs)."
        )
    if not _SINGLE_SAMPLE_QUERY_PATTERN.match(sample_query):
        raise ValueError(
            f"sample_query {sample_query!r} does not select a single "
            "sample. snp_calls_to_vcf() only supports exporting a "
            "single sample at a time (its current use case is "
            "generating a per-sample VCF for IGV); sample_query must be "
            "of the form \"sample_id == '<sample_id>'\"."
        )


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

            `sample_sets` and `sample_query` are both required: a VCF is
            only meaningful with respect to an explicit sample basis for
            its calls. The current use case for this functionality is
            generating a single-sample VCF for IGV, so `sample_query` is
            required to select exactly one sample, and must be of the
            form `"sample_id == '<sample_id>'"`. Any other query (e.g.
            selecting samples by cohort, or more than one sample_id) is
            rejected.
        """,
        returns="""
        Path to the VCF output file.
        """,
    )
    def snp_calls_to_vcf(
        self,
        output_path: vcf_params.vcf_output_path,
        region: base_params.regions,
        sample_sets: base_params.sample_sets,
        sample_query: base_params.sample_query,
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
        _validate_single_sample_selection(
            sample_sets=sample_sets, sample_query=sample_query
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

        # Canonicalise the field order to _FIELD_ORDER. The per-sample
        # values below are always written in this fixed order; if the
        # FORMAT column (built from `fields` further down) used the
        # caller's order instead, a `fields` argument with a different
        # (or undefined, e.g. a set) order would produce a VCF whose
        # FORMAT column lies about which value is which — which is
        # exactly what happened before this fix.
        fields = tuple(f for f in _FIELD_ORDER if f in fields)

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

        # Confirm inputs actually resolved to exactly
        # one sample within the given sample_sets
        # checked defensively rather than silently written
        # out as a multi-sample VCF.
        n_samples_selected = ds.sizes["samples"]
        if n_samples_selected != 1:
            raise ValueError(
                f"sample_query {sample_query!r} selected "
                f"{n_samples_selected} samples from sample_sets "
                f"{sample_sets!r}, expected exactly 1."
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

            optional_arrays = self._get_optional_data(
                include_gq, include_ad, include_mq, ds
            )

            chunk_sizes = gt_data.chunks[0]
            offsets = np.cumsum((0,) + chunk_sizes)

            # Write records in chunks.
            with self._spinner(f"Write VCF ({ds.sizes['variants']} variants)"):
                for ci in range(len(chunk_sizes)):
                    variant_chunk_data = self._get_chunks(
                        ci,
                        offsets,
                        gt_data,
                        pos_data,
                        contig_data,
                        allele_data,
                        optional_arrays,
                    )

                    # OPTIMIZATION: Vectorize GT field formatting across entire chunk.
                    # Instead of formatting each sample's GT field in a nested Python loop
                    # (which results in billions of string operations for large datasets),
                    # use NumPy's vectorized string operations on the entire chunk at once.
                    # This provides ~3x speedup while maintaining exact output compatibility.
                    # See issue #1280 for performance analysis.
                    gt_chunk_2d = variant_chunk_data.gt_chunk[:, 0, :]
                    a0 = gt_chunk_2d[:, 0]  # n_variants
                    a1 = gt_chunk_2d[:, 1]  # (n_variants
                    missing = (a0 < 0) | (a1 < 0)

                    # Build formatted GT strings using NumPy vectorization
                    gt_formatted = np.empty(
                        variant_chunk_data.gt_chunk.shape[0], dtype=object
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

                    for j in range(variant_chunk_data.gt_chunk.shape[0]):
                        chrom = contigs[variant_chunk_data.contig_chunk[j]]
                        pos = str(variant_chunk_data.pos_chunk[j])
                        alleles = variant_chunk_data.allele_chunk[j]
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

                        genotype = gt_formatted[j]
                        if genotype == "0/0":
                            continue

                        parts = [genotype]
                        # GQ.
                        if include_gq:
                            if variant_chunk_data.gq_chunk is not None:
                                value = variant_chunk_data.gq_chunk[j, 0]
                                parts.append("." if value < 0 else str(value))
                            else:
                                parts.append(".")
                        # AD.
                        if include_ad:
                            if variant_chunk_data.ad_chunk is not None:
                                ad_values = variant_chunk_data.ad_chunk[j, 0]
                                parts.append(
                                    ",".join(
                                        "." if value < 0 else str(value)
                                        for value in ad_values
                                    )
                                )
                            else:
                                parts.append(".")
                        # MQ. Rounded to the nearest integer: the
                        # underlying data is a float, but the VCF spec
                        # fixes the reserved FORMAT/MQ key as Integer.
                        if include_mq:
                            if variant_chunk_data.mq_chunk is not None:
                                value = variant_chunk_data.mq_chunk[j, 0]
                                if not math.isnan(value):
                                    parts.append(
                                        "." if value < 0 else str(round(value))
                                    )
                            else:
                                parts.append(".")

                        line = fixed_cols + ":".join(parts) + "\n"
                        lines_to_write.append(line)

                    # Write buffered lines in one go per chunk
                    f.write("".join(lines_to_write))

        return output_path

    def _get_chunks(
        self,
        ci: int,
        offsets: np.array,
        gt_data: Array,
        pos_data: Array,
        contig_data: Array,
        allele_data: Array,
        optional_arrays: dict,
    ) -> tuple:
        start = offsets[ci]
        stop = offsets[ci + 1]

        # Fetch the required arrays for this chunk in a
        # single batched call, rather than one .compute()
        # call per array. Each separate .compute() call is
        # its own blocking round trip to the underlying
        # store (e.g. GCS); batching lets dask fetch all of
        # them concurrently instead of one at a time
        gt_chunk, pos_chunk, contig_chunk, allele_chunk = dask.compute(
            gt_data[start:stop],
            pos_data[start:stop],
            contig_data[start:stop],
            allele_data[start:stop],
        )
        # Fetch requested optional fields for this chunk
        # as a single batched call. If any of them fail
        # to load, fall back to "." for all requested
        # optional fields in this chunk — these arrays live
        # in the same region of the same store, so a failure
        # affecting one is likely to affect the others too,
        gq_chunk = None
        ad_chunk = None
        mq_chunk = None
        if optional_arrays:
            try:
                computed = dask.compute(
                    *(arr[start:stop] for arr in optional_arrays.values())
                )
                optional_chunks = dict(zip(optional_arrays.keys(), computed))
                gq_chunk = optional_chunks.get("GQ")
                ad_chunk = optional_chunks.get("AD")
                mq_chunk = optional_chunks.get("MQ")
            except (FileNotFoundError, KeyError):
                pass
        chunk_output = VariantChunkData(
            gt_chunk=gt_chunk,
            pos_chunk=pos_chunk,
            contig_chunk=contig_chunk,
            allele_chunk=allele_chunk,
            gq_chunk=gq_chunk,
            ad_chunk=ad_chunk,
            mq_chunk=mq_chunk,
        )
        return chunk_output

    def _get_optional_data(
        self, include_gq: bool, include_ad: bool, include_mq: bool, ds: Dataset
    ) -> dict:
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
        # Which optional fields were requested, so we know what to
        # fetch (and where to put the results) for each chunk below.
        optional_arrays = {}
        if gq_data is not None:
            optional_arrays["GQ"] = gq_data
        if ad_data is not None:
            optional_arrays["AD"] = ad_data
        if mq_data is not None:
            optional_arrays["MQ"] = mq_data
        return optional_arrays
