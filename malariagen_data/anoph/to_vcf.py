import gzip
import os
import re
from datetime import date
from typing import Optional

import dask
import numpy as np
from numpydoc_decorator import doc  # type: ignore

from .snp_data import AnophelesSnpData
from . import base_params
from . import plink_params
from . import vcf_params

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
        parameters=dict(
            non_ref_only="""
                If True, only write sites where the sample's genotype
                carries at least one non-reference allele (i.e. skip
                sites where the sample is homozygous reference or
                missing). Equivalent to filtering the output with
                `bcftools view -e 'F_PASS(GT="ref") == 1'`, except that
                missing genotype calls are also excluded here (whereas
                that particular bcftools expression only excludes
                homozygous reference calls, since a missing call is
                neither "ref" nor selected by `-e`).
            """,
        ),
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
        site_mask: Optional[base_params.site_mask] = None,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
        overwrite: plink_params.overwrite = False,
        fields: vcf_params.vcf_fields = ("GT",),
        non_ref_only: bool = False,
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

            # Which optional fields were requested, so we know what to
            # fetch (and where to put the results) for each chunk below.
            optional_names = []
            optional_arrays = []
            if gq_data is not None:
                optional_names.append("GQ")
                optional_arrays.append(gq_data)
            if ad_data is not None:
                optional_names.append("AD")
                optional_arrays.append(ad_data)
            if mq_data is not None:
                optional_names.append("MQ")
                optional_arrays.append(mq_data)

            # Write records in chunks.
            with self._spinner(f"Write VCF ({ds.sizes['variants']} variants)"):
                for ci in range(len(chunk_sizes)):
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
                                *(arr[start:stop] for arr in optional_arrays)
                            )
                            optional_chunks = dict(zip(optional_names, computed))
                            gq_chunk = optional_chunks.get("GQ")
                            ad_chunk = optional_chunks.get("AD")
                            mq_chunk = optional_chunks.get("MQ")
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

                    # If non_ref_only, work out up front which variants
                    # to skip: those where the sample is homozygous
                    # reference (0/0) or missing. Computed vectorized
                    # across the whole chunk, same as `missing` above.
                    # N.B., this uses the *original* (un-remapped)
                    # allele indices computed below — 0 always means REF
                    # both before and after remapping, so this check is
                    # unaffected by it.
                    if non_ref_only:
                        is_hom_ref = (a0 == 0) & (a1 == 0)
                        skip_variant = (missing | is_hom_ref)[:, 0]
                    else:
                        skip_variant = None

                    # Decode alleles once per chunk (rather than
                    # per-variant further down) and work out, per
                    # variant, which of the ALT candidate slots are
                    # actually populated. call_genotype's allele indices
                    # refer to these *original* variant_allele slot
                    # positions (0=REF, 1..3=ALT candidates), which are
                    # not guaranteed to be contiguous from 1 — a site's
                    # only real ALT allele can sit in slot 2 or 3 while
                    # an earlier slot is empty. The ALT column written
                    # below only lists the populated slots, so genotype
                    # indices must be remapped to match that compacted
                    # numbering, or a GT value could end up referring to
                    # an ALT allele that isn't listed at all (or the
                    # wrong one).
                    decoded_alleles = np.empty(allele_chunk.shape, dtype=object)
                    for col in range(allele_chunk.shape[1]):
                        decoded_alleles[:, col] = [
                            a.decode() if hasattr(a, "decode") else str(a)
                            for a in allele_chunk[:, col]
                        ]
                    is_present = decoded_alleles != ""
                    is_present[:, 0] = True  # REF is always present
                    # compacted_index[i, k] = the index original slot k
                    # occupies in the ALT column for variant i, or -1 if
                    # slot k isn't populated there.
                    compacted_index = np.cumsum(is_present, axis=1) - 1
                    compacted_index[~is_present] = -1

                    # Remap this chunk's genotype allele indices to the
                    # compacted numbering. n_samples is always 1 here
                    # (snp_calls_to_vcf only supports a single sample),
                    # so a0/a1 are worked with as flat (n_variants,)
                    # arrays. Clip before gathering so a missing
                    # genotype's sentinel value (-1) can't trigger an
                    # out-of-bounds index — the gathered result for
                    # those rows is discarded below anyway, since they
                    # are written as "./." based on `missing`, not on
                    # this remapping.
                    row_idx = np.arange(gt_chunk.shape[0])
                    n_allele_slots = allele_chunk.shape[1]
                    remapped_a0 = compacted_index[
                        row_idx, np.clip(a0[:, 0], 0, n_allele_slots - 1)
                    ]
                    remapped_a1 = compacted_index[
                        row_idx, np.clip(a1[:, 0], 0, n_allele_slots - 1)
                    ]
                    # A remapped index of -1 means the genotype
                    # references an allele slot that variant_allele
                    # doesn't actually have populated at this site —
                    # inconsistent data that shouldn't occur, but is
                    # treated as missing defensively rather than writing
                    # a nonsensical negative allele index.
                    missing_or_unmapped = (
                        missing[:, 0] | (remapped_a0 < 0) | (remapped_a1 < 0)
                    )

                    # Build formatted GT strings using NumPy vectorization
                    gt_formatted = np.empty(
                        (gt_chunk.shape[0], n_samples), dtype=object
                    )
                    gt_formatted[missing_or_unmapped, 0] = "./."
                    present_idx = ~missing_or_unmapped
                    if np.any(present_idx):
                        a0_str = remapped_a0[present_idx].astype(str)
                        a1_str = remapped_a1[present_idx].astype(str)
                        gt_formatted[present_idx, 0] = np.char.add(
                            np.char.add(a0_str, "/"), a1_str
                        )

                    # Pre-allocate line buffer for better I/O
                    lines_to_write = []

                    for j in range(gt_chunk.shape[0]):
                        if skip_variant is not None and skip_variant[j]:
                            continue

                        chrom = contigs[contig_chunk[j]]
                        pos = str(pos_chunk[j])
                        # Reuse the chunk-level decode from above, so
                        # the ALT column built here is guaranteed
                        # consistent with the allele-index remapping
                        # used for GT (both derive from the same
                        # decoded_alleles/is_present arrays).
                        ref = decoded_alleles[j, 0]
                        alt_alleles = [s for s in decoded_alleles[j, 1:] if s]
                        alt = ",".join(alt_alleles) if alt_alleles else "."

                        # Build fixed VCF columns once per variant
                        fixed_cols = (
                            f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\t{format_str}\t"
                        )

                        # N.B., a plain Python list here, not a NumPy
                        # array: this loop builds a handful of short
                        # strings per variant via ordinary Python string
                        # concatenation, not a numeric/vectorized
                        # operation, so a `np.empty(..., dtype=object)`
                        # array bought nothing but the overhead of a
                        # NumPy allocation on every single variant row
                        # (up to ~150 million times for a whole-genome
                        # export).
                        sample_fields = []

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
                            # MQ. Rounded to the nearest integer: the
                            # underlying data is a float, but the VCF
                            # spec fixes the reserved FORMAT/MQ key as
                            # Integer (see _FORMAT_HEADERS).
                            if include_mq:
                                if mq_chunk is not None:
                                    v = mq_chunk[j, k]
                                    parts.append("." if v < 0 else str(round(v)))
                                else:
                                    parts.append(".")
                            sample_fields.append(":".join(parts))

                        # Build and buffer the line
                        line = fixed_cols + "\t".join(sample_fields) + "\n"
                        lines_to_write.append(line)

                    # Write buffered lines in one go per chunk
                    f.write("".join(lines_to_write))

        return output_path
