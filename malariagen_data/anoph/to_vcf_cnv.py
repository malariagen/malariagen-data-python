import gzip
import os
from datetime import date
from typing import List, Optional, Sequence

import numpy as np
from numpydoc_decorator import doc  # type: ignore

from .cnv_data import AnophelesCnvData
from . import base_params
from . import cnv_params
from . import plink_params
from . import vcf_params


def _decode(value) -> str:
    return value.decode() if hasattr(value, "decode") else str(value)


def _write_vcf_header(
    f,
    *,
    contigs: Sequence[str],
    sample_ids: Sequence[str],
    info_lines: Sequence[str],
    format_lines: Sequence[str],
    alt_lines: Sequence[str] = (),
) -> None:
    f.write("##fileformat=VCFv4.3\n")
    f.write(f"##fileDate={date.today().strftime('%Y%m%d')}\n")
    f.write("##source=malariagen_data\n")
    for contig in contigs:
        f.write(f"##contig=<ID={contig}>\n")
    for line in alt_lines:
        f.write(line + "\n")
    for line in info_lines:
        f.write(line + "\n")
    for line in format_lines:
        f.write(line + "\n")
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


class CnvVcfExporter(
    AnophelesCnvData,
):
    """
    Export CNV calls to VCF.

    N.B., unlike SNP calls, CNV calls are not fully standardised in VCF.
    Here each CNV region or call is written as a single symbolic `<CNV>`
    ALT allele (REF is written as "N", per the VCF specification for
    imprecise/symbolic variants), with the relevant per-sample data
    encoded in FORMAT fields. This is a draft representation intended
    to give IGV something sensible to render, and to remove the
    dependency on separately-hosted CNV VCF files; it is not intended
    to be a fully general-purpose CNV VCF writer.
    """

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
            Export CNV HMM copy number calls to Variant Call Format (VCF).
        """,
        extended_summary="""
            This function writes genome-wide CNV HMM calls to a VCF file.
            Each CNV region is written as a symbolic `<CNV>` allele, with
            the estimated copy number for each sample stored in the `CN`
            FORMAT field. Data is written in chunks to avoid loading the
            entire array into memory. Supports optional gzip compression
            when the output path ends with `.gz`.
        """,
        returns="""
        Path to the VCF output file.
        """,
    )
    def cnv_hmm_to_vcf(
        self,
        output_path: vcf_params.vcf_output_path,
        region: base_params.regions,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        max_coverage_variance: cnv_params.max_coverage_variance = cnv_params.max_coverage_variance_default,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
        overwrite: plink_params.overwrite = False,
    ) -> str:
        if os.path.exists(output_path) and not overwrite:
            return output_path

        ds = self.cnv_hmm(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            max_coverage_variance=max_coverage_variance,
            inline_array=inline_array,
            chunks=chunks,
        )

        sample_ids = ds["sample_id"].values
        contigs = ds.attrs.get("contigs", self.contigs)
        compress = output_path.endswith(".gz")
        opener = gzip.open if compress else open

        with opener(output_path, "wt") as f:
            _write_vcf_header(
                f,
                contigs=contigs,
                sample_ids=sample_ids,
                alt_lines=[
                    '##ALT=<ID=CNV,Description="Copy number variant region">',
                ],
                info_lines=[
                    '##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the CNV region">',
                ],
                format_lines=[
                    '##FORMAT=<ID=CN,Number=1,Type=Integer,Description="Copy number estimated from read coverage by HMM">',
                ],
            )

            pos_data = ds["variant_position"].data
            end_data = ds["variant_end"].data
            contig_data = ds["variant_contig"].data
            cn_data = ds["call_CN"].data

            chunk_sizes = pos_data.chunks[0]
            offsets = np.cumsum((0,) + chunk_sizes)

            with self._spinner(f"Write VCF ({ds.sizes['variants']} CNV HMM regions)"):
                for ci in range(len(chunk_sizes)):
                    start = offsets[ci]
                    stop = offsets[ci + 1]
                    pos_chunk = pos_data[start:stop].compute()
                    end_chunk = end_data[start:stop].compute()
                    contig_chunk = contig_data[start:stop].compute()
                    cn_chunk = cn_data[start:stop].compute()

                    lines: List[str] = []
                    for j in range(pos_chunk.shape[0]):
                        chrom = contigs[contig_chunk[j]]
                        pos = str(pos_chunk[j])
                        end = int(end_chunk[j])
                        info = f"END={end}"
                        sample_fields = (
                            "." if v < 0 else str(int(v)) for v in cn_chunk[j]
                        )
                        lines.append(
                            f"{chrom}\t{pos}\t.\tN\t<CNV>\t.\t.\t{info}\tCN\t"
                            + "\t".join(sample_fields)
                            + "\n"
                        )
                    f.write("".join(lines))

        return output_path

    @doc(
        summary="""
            Export CNV coverage calls to Variant Call Format (VCF).
        """,
        extended_summary="""
            This function writes CNV coverage calls (discrete CNV alleles
            called from read coverage) to a VCF file. Each CNV allele is
            written as a symbolic `<CNV>` allele, with the presence/absence
            call for each sample stored in the `GT` FORMAT field (as a
            haploid call: "1" if the CNV allele was called, "0" if not,
            "." if missing). Data is written in chunks to avoid loading
            the entire array into memory. Supports optional gzip
            compression when the output path ends with `.gz`.
        """,
        returns="""
        Path to the VCF output file.
        """,
    )
    def cnv_coverage_calls_to_vcf(
        self,
        output_path: vcf_params.vcf_output_path,
        region: base_params.regions,
        sample_set: base_params.sample_set,
        analysis: cnv_params.coverage_calls_analysis,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
        overwrite: plink_params.overwrite = False,
    ) -> str:
        if os.path.exists(output_path) and not overwrite:
            return output_path

        ds = self.cnv_coverage_calls(
            region=region,
            sample_set=sample_set,
            analysis=analysis,
            inline_array=inline_array,
            chunks=chunks,
        )

        sample_ids = ds["sample_id"].values
        contigs = ds.attrs.get("contigs", self.contigs)
        compress = output_path.endswith(".gz")
        opener = gzip.open if compress else open

        with opener(output_path, "wt") as f:
            _write_vcf_header(
                f,
                contigs=contigs,
                sample_ids=sample_ids,
                alt_lines=[
                    '##ALT=<ID=CNV,Description="Copy number variant region">',
                ],
                info_lines=[
                    '##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the CNV allele">',
                    '##INFO=<ID=CIPOS,Number=1,Type=Integer,Description="Confidence interval half-width around POS">',
                    '##INFO=<ID=CIEND,Number=1,Type=Integer,Description="Confidence interval half-width around END">',
                ],
                format_lines=[
                    '##FORMAT=<ID=GT,Number=1,Type=String,Description="CNV call (1=called, 0=not called)">',
                ],
            )

            pos_data = ds["variant_position"].data
            end_data = ds["variant_end"].data
            contig_data = ds["variant_contig"].data
            id_data = ds["variant_id"].data
            cipos_data = ds["variant_CIPOS"].data
            ciend_data = ds["variant_CIEND"].data
            filter_data = ds["variant_filter_pass"].data
            gt_data = ds["call_genotype"].data

            chunk_sizes = pos_data.chunks[0]
            offsets = np.cumsum((0,) + chunk_sizes)

            with self._spinner(
                f"Write VCF ({ds.sizes['variants']} CNV coverage calls)"
            ):
                for ci in range(len(chunk_sizes)):
                    start = offsets[ci]
                    stop = offsets[ci + 1]
                    pos_chunk = pos_data[start:stop].compute()
                    end_chunk = end_data[start:stop].compute()
                    contig_chunk = contig_data[start:stop].compute()
                    id_chunk = id_data[start:stop].compute()
                    cipos_chunk = cipos_data[start:stop].compute()
                    ciend_chunk = ciend_data[start:stop].compute()
                    filter_chunk = filter_data[start:stop].compute()
                    gt_chunk = gt_data[start:stop].compute()

                    lines: List[str] = []
                    for j in range(pos_chunk.shape[0]):
                        chrom = contigs[contig_chunk[j]]
                        pos = str(pos_chunk[j])
                        variant_id = _decode(id_chunk[j])
                        end = int(end_chunk[j])
                        filt = "PASS" if filter_chunk[j] else "."
                        info = (
                            f"END={end};CIPOS={int(cipos_chunk[j])}"
                            f";CIEND={int(ciend_chunk[j])}"
                        )
                        sample_fields = (
                            "." if v < 0 else str(int(v)) for v in gt_chunk[j]
                        )
                        lines.append(
                            f"{chrom}\t{pos}\t{variant_id}\tN\t<CNV>\t.\t{filt}\t{info}\tGT\t"
                            + "\t".join(sample_fields)
                            + "\n"
                        )
                    f.write("".join(lines))

        return output_path

    @doc(
        summary="""
            Export CNV discordant read calls to Variant Call Format (VCF).
        """,
        extended_summary="""
            This function writes CNV discordant read calls (CNV alleles
            supported by discordant/paired-end read evidence) to a VCF
            file. Each CNV allele is written as a symbolic `<CNV>` allele,
            with the call for each sample stored in the `GT` FORMAT field
            (as a haploid call: "1" if the CNV allele was called, "0" if
            not, "." if missing). Data is written in chunks to avoid
            loading the entire array into memory. Supports optional gzip
            compression when the output path ends with `.gz`.
        """,
        returns="""
        Path to the VCF output file.
        """,
    )
    def cnv_discordant_read_calls_to_vcf(
        self,
        output_path: vcf_params.vcf_output_path,
        contigs: base_params.contigs,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
        overwrite: plink_params.overwrite = False,
    ) -> str:
        if os.path.exists(output_path) and not overwrite:
            return output_path

        ds = self.cnv_discordant_read_calls(
            contigs=contigs,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            inline_array=inline_array,
            chunks=chunks,
        )

        sample_ids = ds["sample_id"].values
        all_contigs = ds.attrs.get("contigs", self.contigs)
        compress = output_path.endswith(".gz")
        opener = gzip.open if compress else open

        with opener(output_path, "wt") as f:
            _write_vcf_header(
                f,
                contigs=all_contigs,
                sample_ids=sample_ids,
                alt_lines=[
                    '##ALT=<ID=CNV,Description="Copy number variant region">',
                ],
                info_lines=[
                    '##INFO=<ID=END,Number=1,Type=Integer,Description="End position of the CNV allele">',
                    '##INFO=<ID=REGION,Number=1,Type=String,Description="Identifier of the region covered by this call">',
                    '##INFO=<ID=SBM,Number=1,Type=Integer,Description="Method used to determine the start breakpoint">',
                    '##INFO=<ID=EBM,Number=1,Type=Integer,Description="Method used to determine the end breakpoint">',
                ],
                format_lines=[
                    '##FORMAT=<ID=GT,Number=1,Type=String,Description="CNV call (1=called, 0=not called)">',
                ],
            )

            pos_data = ds["variant_position"].data
            end_data = ds["variant_end"].data
            contig_data = ds["variant_contig"].data
            id_data = ds["variant_id"].data
            region_data = ds["variant_Region"].data
            sbm_data = ds["variant_StartBreakpointMethod"].data
            ebm_data = ds["variant_EndBreakpointMethod"].data
            gt_data = ds["call_genotype"].data

            chunk_sizes = pos_data.chunks[0]
            offsets = np.cumsum((0,) + chunk_sizes)

            with self._spinner(
                f"Write VCF ({ds.sizes['variants']} CNV discordant read calls)"
            ):
                for ci in range(len(chunk_sizes)):
                    start = offsets[ci]
                    stop = offsets[ci + 1]
                    pos_chunk = pos_data[start:stop].compute()
                    end_chunk = end_data[start:stop].compute()
                    contig_chunk = contig_data[start:stop].compute()
                    id_chunk = id_data[start:stop].compute()
                    region_chunk = region_data[start:stop].compute()
                    sbm_chunk = sbm_data[start:stop].compute()
                    ebm_chunk = ebm_data[start:stop].compute()
                    gt_chunk = gt_data[start:stop].compute()

                    lines: List[str] = []
                    for j in range(pos_chunk.shape[0]):
                        chrom = all_contigs[contig_chunk[j]]
                        pos = str(pos_chunk[j])
                        variant_id = _decode(id_chunk[j])
                        end = int(end_chunk[j])
                        region_val = _decode(region_chunk[j])
                        info = (
                            f"END={end};REGION={region_val}"
                            f";SBM={int(sbm_chunk[j])};EBM={int(ebm_chunk[j])}"
                        )
                        sample_fields = (
                            "." if v < 0 else str(int(v)) for v in gt_chunk[j]
                        )
                        lines.append(
                            f"{chrom}\t{pos}\t{variant_id}\tN\t<CNV>\t.\t.\t{info}\tGT\t"
                            + "\t".join(sample_fields)
                            + "\n"
                        )
                    f.write("".join(lines))

        return output_path
