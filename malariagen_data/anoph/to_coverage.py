import gzip
import os
from typing import List, Optional

import numpy as np
from numpydoc_decorator import doc  # type: ignore

from .cnv_data import AnophelesCnvData
from .snp_data import AnophelesSnpData
from . import base_params
from . import cnv_params
from . import plink_params

_VALID_CNV_COVERAGE_FIELDS = {"NormCov", "RawCov"}


def _bedgraph_opener(output_path: str):
    # N.B., plain gzip is intentional here, unlike the VCF exporters
    # (see malariagen_data.bgzf): bedGraph doesn't carry the same
    # ecosystem-wide expectation that a `.gz` file specifically be
    # BGZF, and IGV loads a plain-gzipped bedGraph track without
    # issue.
    compress = output_path.endswith(".gz")
    return gzip.open if compress else open


class CoverageExporter(
    AnophelesCnvData,
    AnophelesSnpData,
):
    """
    Export per-sample read coverage to bedGraph, derived either from
    CNV HMM data (windowed coverage, genome-wide) or from SNP calls
    (per-site read depth, only at called SNP positions).

    N.B., this does not reconstruct read alignments. It cannot
    substitute for a BAM file, because the underlying Zarr arrays only
    ever store windowed or per-site summary values (and copy number
    calls derived from them) rather than individual reads. What this
    class provides is a coverage-depth track that can be loaded into
    IGV alongside (or instead of) an alignments track, as a fallback
    for when the raw BAM file for a sample is not available, e.g.
    because it is hosted externally and that host has become
    unreachable.
    """

    def __init__(
        self,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

    def _lookup_sample_set(self, sample: str) -> str:
        try:
            sample_rec = self.sample_metadata().set_index("sample_id").loc[sample]
        except KeyError as e:
            raise ValueError(
                f"No data found for sample {sample!r}. This sample might be "
                "unavailable or irrelevant with respect to settings."
            ) from e
        return sample_rec["sample_set"]

    @doc(
        summary="""
            Export per-sample CNV HMM coverage to bedGraph format.
        """,
        extended_summary="""
            This function writes windowed read coverage for a single
            sample, derived from CNV HMM data, to a bedGraph file. This
            is not a substitute for a BAM file of aligned reads: it
            contains only per-window summary coverage values (as stored
            in the Zarr arrays), not individual reads. It is intended as
            a fallback coverage track for IGV, for use when the aligned
            reads (BAM) for a sample are not available, e.g. because the
            file is hosted externally and unreachable. Coverage is
            available genome-wide (in fixed-size windows), regardless of
            whether any SNPs were called nearby. Data is written in
            chunks to avoid loading the entire array into memory.
            Supports optional gzip compression when the output path ends
            with `.gz`.
        """,
        parameters=dict(
            output_path="""
                Path to write the bedGraph output file. Use a `.gz`
                extension to enable gzip compression.
            """,
            sample="Sample identifier.",
            field="""
                Which coverage field to export. "NormCov" is coverage
                normalised against a set of reference windows (comparable
                between samples); "RawCov" is unnormalised read counts
                per window.
            """,
        ),
        returns="""
        Path to the bedGraph output file.
        """,
    )
    def cnv_hmm_coverage_to_bedgraph(
        self,
        output_path: str,
        region: base_params.regions,
        sample: str,
        field: str = "NormCov",
        max_coverage_variance: cnv_params.max_coverage_variance = None,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
        overwrite: plink_params.overwrite = False,
    ) -> str:
        if field not in _VALID_CNV_COVERAGE_FIELDS:
            raise ValueError(
                f"Unknown coverage field: {field!r}. "
                f"Valid fields are: {sorted(_VALID_CNV_COVERAGE_FIELDS)}"
            )

        if os.path.exists(output_path) and not overwrite:
            return output_path

        # Look up which sample set this sample belongs to, so we only
        # need to open the CNV HMM data for that one sample set.
        sample_set = self._lookup_sample_set(sample)

        ds = self.cnv_hmm(
            region=region,
            sample_sets=sample_set,
            sample_query=f"sample_id == {sample!r}",
            max_coverage_variance=max_coverage_variance,
            inline_array=inline_array,
            chunks=chunks,
        )

        if ds.sizes["samples"] == 0:
            raise ValueError(
                f"No CNV HMM data available for sample {sample!r}, this may be "
                "because it was excluded by max_coverage_variance."
            )

        contigs = ds.attrs.get("contigs", self.contigs)
        opener = _bedgraph_opener(output_path)

        with opener(output_path, "wt") as f:
            f.write(
                f'track type=bedGraph name="{sample} {field}" '
                f'description="CNV HMM {field} for {sample}"\n'
            )

            pos_data = ds["variant_position"].data
            end_data = ds["variant_end"].data
            contig_data = ds["variant_contig"].data
            cov_data = ds[f"call_{field}"].data

            chunk_sizes = pos_data.chunks[0]
            offsets = np.cumsum((0,) + chunk_sizes)

            with self._spinner(
                f"Write bedGraph ({ds.sizes['variants']} coverage windows)"
            ):
                for ci in range(len(chunk_sizes)):
                    start = offsets[ci]
                    stop = offsets[ci + 1]
                    pos_chunk = pos_data[start:stop].compute()
                    end_chunk = end_data[start:stop].compute()
                    contig_chunk = contig_data[start:stop].compute()
                    # Select the single requested sample's coverage
                    # column from the already-materialised chunk, rather
                    # than indexing the dask/zarr array before compute,
                    # which can trigger a less efficient (and, for
                    # arrays with unwritten/fill-value chunks, buggy)
                    # orthogonal-selection code path.
                    cov_chunk = cov_data[start:stop].compute()[:, 0]

                    lines: List[str] = []
                    for j in range(pos_chunk.shape[0]):
                        v = cov_chunk[j]
                        if v < 0:
                            # Masked/missing value, skip rather than
                            # write a misleading zero.
                            continue
                        chrom = contigs[contig_chunk[j]]
                        # bedGraph coordinates are 0-based, half-open;
                        # variant_position is 1-based inclusive.
                        bed_start = int(pos_chunk[j]) - 1
                        bed_end = int(end_chunk[j])
                        lines.append(f"{chrom}\t{bed_start}\t{bed_end}\t{v}\n")
                    f.write("".join(lines))

        return output_path

    @doc(
        summary="""
            Export per-sample SNP read depth to bedGraph format.
        """,
        extended_summary="""
            This function writes read depth at called SNP sites for a
            single sample, derived from the allele depth (`AD`) field of
            SNP calls, to a bedGraph file. Unlike
            `cnv_hmm_coverage_to_bedgraph()`, this only has a value at
            positions where a SNP was called, not genome-wide, but gives
            higher-resolution, per-base depth in those locations rather
            than a windowed summary. This is not a substitute for a BAM
            file of aligned reads. It is intended as a fallback coverage
            track for IGV, for use when the aligned reads (BAM) for a
            sample are not available, e.g. because the file is hosted
            externally and unreachable. Data is written in chunks to
            avoid loading the entire array into memory. Supports
            optional gzip compression when the output path ends with
            `.gz`.
        """,
        parameters=dict(
            output_path="""
                Path to write the bedGraph output file. Use a `.gz`
                extension to enable gzip compression.
            """,
            sample="Sample identifier.",
        ),
        returns="""
        Path to the bedGraph output file.
        """,
    )
    def snp_calls_coverage_to_bedgraph(
        self,
        output_path: str,
        region: base_params.regions,
        sample: str,
        site_mask: Optional[base_params.site_mask] = None,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        chunks: base_params.chunks = base_params.native_chunks,
        overwrite: plink_params.overwrite = False,
    ) -> str:
        if os.path.exists(output_path) and not overwrite:
            return output_path

        # Look up which sample set this sample belongs to, so we only
        # need to open the SNP calls data for that one sample set.
        sample_set = self._lookup_sample_set(sample)

        ds = self.snp_calls(
            region=region,
            sample_sets=sample_set,
            sample_query=f"sample_id == {sample!r}",
            site_mask=site_mask,
            inline_array=inline_array,
            chunks=chunks,
        )

        if ds.sizes["samples"] == 0:
            raise ValueError(f"No SNP calls data available for sample {sample!r}.")

        contigs = ds.attrs.get("contigs", self.contigs)
        opener = _bedgraph_opener(output_path)

        with opener(output_path, "wt") as f:
            f.write(
                f'track type=bedGraph name="{sample} SNP depth" '
                f'description="SNP read depth (sum of AD) for {sample}"\n'
            )

            pos_data = ds["variant_position"].data
            contig_data = ds["variant_contig"].data
            ad_data = ds["call_AD"].data

            chunk_sizes = pos_data.chunks[0]
            offsets = np.cumsum((0,) + chunk_sizes)

            with self._spinner(f"Write bedGraph ({ds.sizes['variants']} SNP sites)"):
                for ci in range(len(chunk_sizes)):
                    start = offsets[ci]
                    stop = offsets[ci + 1]
                    pos_chunk = pos_data[start:stop].compute()
                    contig_chunk = contig_data[start:stop].compute()
                    # Select the single requested sample's allele depth
                    # values from the already-materialised chunk, rather
                    # than indexing the dask/zarr array before compute
                    # (see the equivalent comment in
                    # cnv_hmm_coverage_to_bedgraph).
                    ad_chunk = ad_data[start:stop].compute()[:, 0, :]

                    lines: List[str] = []
                    for j in range(pos_chunk.shape[0]):
                        ad_row = ad_chunk[j]
                        if np.all(ad_row < 0):
                            # No call for this sample at this site, skip
                            # rather than write a misleading zero.
                            continue
                        depth = int(np.clip(ad_row, 0, None).sum())
                        chrom = contigs[contig_chunk[j]]
                        # bedGraph coordinates are 0-based, half-open;
                        # variant_position is 1-based, and each SNP is a
                        # single base position.
                        bed_start = int(pos_chunk[j]) - 1
                        bed_end = int(pos_chunk[j])
                        lines.append(f"{chrom}\t{bed_start}\t{bed_end}\t{depth}\n")
                    f.write("".join(lines))

        return output_path
