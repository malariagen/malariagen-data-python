import os
from typing import List, Optional

import igv_notebook  # type: ignore
from numpydoc_decorator import doc  # type: ignore

from ..util import Region, _check_types, _parse_single_region
from . import base_params
from .snp_data import AnophelesSnpData


class AnophelesIgv(
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

    def _igv_config(
        self,
        *,
        region: Region,
        tracks: Optional[List] = None,
    ):
        # Create IGV config.
        config = {
            "reference": {
                "id": self._genome_ref_id,
                "name": self._genome_ref_name,
                "fastaURL": f"{self._public_url}{self._genome_fasta_path}",
                "indexURL": f"{self._public_url}{self._genome_fai_path}",
                "tracks": [
                    {
                        "name": "Genes",
                        "type": "annotation",
                        "format": "gff3",
                        "url": f"{self._public_url}{self._geneset_gff3_path}",
                        "indexed": False,
                    }
                ],
            },
            "locus": str(region),
        }

        if tracks:
            config["tracks"] = tracks

        return config

    def _igv_site_filters_tracks(
        self,
        *,
        contig,
        visibility_window,
    ):
        tracks = []
        for site_mask in self.site_mask_ids:
            site_filters_vcf_url = f"{self._public_url}{self._major_version_path}/site_filters/{self._site_filters_analysis}/vcf/{site_mask}/{contig}_sitefilters.vcf.gz"  # f"{self._url}{self._major_version_path}/site_filters/{self._site_filters_analysis}/vcf/{site_mask}/{contig}_sitefilters.vcf.gz"  # noqa
            track_config = {
                "name": f"Filters - {site_mask}",
                "url": site_filters_vcf_url,
                "indexURL": f"{site_filters_vcf_url}.tbi",
                "format": "vcf",
                "type": "variant",
                "visibilityWindow": visibility_window,  # bp
                "height": 30,
                "colorBy": "FILTER",
                "colorTable": {
                    "PASS": "#00cc96",
                    "*": "#ef553b",
                },
            }
            tracks.append(track_config)
        return tracks

    def _igv_local_coverage_track(
        self,
        *,
        region: Region,
        sample: str,
        visibility_window: int,
        source: str = "cnv_hmm",
        field: str = "NormCov",
        output_path: Optional[str] = None,
    ):
        # Build a coverage track computed locally from the Zarr arrays
        # rather than loaded from an externally-hosted file. This is not
        # a substitute for an alignments track (there are no individual
        # reads), but gives a fallback source of coverage information
        # for a sample when the BAM file is not available. Two sources
        # are supported: "cnv_hmm" gives windowed coverage genome-wide;
        # "snp_calls" gives per-base read depth, but only at called SNP
        # sites.
        #
        # N.B., this method is only usable when this class is combined
        # (via cooperative multiple inheritance) with CoverageExporter,
        # as is the case for the top-level Ag3/Af1 data resource classes.
        # Methods are accessed via getattr rather than a direct import to
        # avoid adding CoverageExporter as a base class of AnophelesIgv,
        # which would conflict with the method resolution order of the
        # combined data resource classes.
        extra_kwargs: dict
        if source == "cnv_hmm":
            to_bedgraph = getattr(self, "cnv_hmm_coverage_to_bedgraph", None)
            extra_kwargs = dict(field=field)
            track_name = f"Coverage ({field}, local) - {sample}"
            path_suffix = field
        elif source == "snp_calls":
            to_bedgraph = getattr(self, "snp_calls_coverage_to_bedgraph", None)
            extra_kwargs = dict()
            track_name = f"SNP read depth (local) - {sample}"
            path_suffix = "AD"
        else:
            raise ValueError(
                f"Unknown coverage source: {source!r}. "
                "Valid sources are 'cnv_hmm', 'snp_calls'."
            )

        if to_bedgraph is None:
            raise NotImplementedError(
                "Local coverage tracks require this API object to also "
                "provide CNV and SNP data (see CoverageExporter)."
            )

        if output_path is None:
            if self._results_cache is None:
                raise ValueError(
                    "Cannot create a local coverage track without knowing "
                    "where to write it. Either pass `coverage_track_path`, "
                    "or configure `results_cache` when creating this API "
                    "object."
                )
            safe_region = (
                str(region).replace(":", "_").replace(",", "").replace("-", "_")
            )
            cache_dir = self._results_cache / "igv_coverage"
            os.makedirs(cache_dir, exist_ok=True)
            output_path = str(
                cache_dir / f"{sample}_{safe_region}_{source}_{path_suffix}.bedgraph"
            )

        to_bedgraph(
            output_path=output_path,
            region=region,
            sample=sample,
            **extra_kwargs,
        )

        return {
            "name": track_name,
            "url": output_path,
            "format": "bedgraph",
            "type": "wig",
            "visibilityWindow": visibility_window,  # bp
            "height": 50,
            "color": "#5e3c99",
        }

    def _igv_view_alignments_tracks(
        self,
        region: Region,
        sample: str,
        visibility_window: int = 20_000,
        local_coverage_fallback: bool = False,
        coverage_source: str = "cnv_hmm",
        coverage_track_path: Optional[str] = None,
    ):
        # Look up sample set for sample.
        try:
            sample_rec = self.sample_metadata().set_index("sample_id").loc[sample]
        except KeyError as e:
            raise ValueError(
                f"No data found for sample {sample!r}. This sample might be unavailable or irrelevant with respect to settings."
            ) from e

        sample_set = sample_rec["sample_set"]

        # Load data catalog.
        df_cat = self.wgs_data_catalog(sample_set=sample_set)

        # Locate record for sample.
        cat_rec = df_cat.set_index("sample_id").loc[sample]
        bam_url = cat_rec["alignments_bam"]
        vcf_url = cat_rec["snp_genotypes_vcf"]

        # Set up site filters tracks.
        contig = region.contig
        tracks = self._igv_site_filters_tracks(
            contig=contig,
            visibility_window=visibility_window,
        )

        # Add SNPs track.
        tracks.append(
            {
                "name": "SNPs",
                "url": vcf_url,
                "indexURL": f"{vcf_url}.tbi",
                "format": "vcf",
                "type": "variant",
                "visibilityWindow": visibility_window,  # bp
                "height": 50,
            }
        )

        # Add alignments track. N.B., this depends on the BAM file being
        # hosted at `bam_url` and remaining reachable; if that URL stops
        # working (e.g. following migration of raw data to ENA/EVA) this
        # track will fail to load in the browser.
        tracks.append(
            {
                "name": "Alignments",
                "url": bam_url,
                "indexURL": f"{bam_url}.bai",
                "format": "bam",
                "type": "alignment",
                "visibilityWindow": visibility_window,  # bp
                "height": 500,
            }
        )

        # Optionally also add a coverage track computed locally from CNV
        # HMM or SNP calls data, as a fallback source of coverage
        # information that does not depend on the alignments BAM file
        # remaining available.
        if local_coverage_fallback:
            tracks.append(
                self._igv_local_coverage_track(
                    region=region,
                    sample=sample,
                    visibility_window=visibility_window,
                    source=coverage_source,
                    output_path=coverage_track_path,
                )
            )

        return tracks

    @_check_types
    @doc(
        summary="Create an IGV browser and inject into the current notebook.",
        parameters=dict(
            tracks="Configuration for any additional tracks.",
            init="If True, call igv_notebook.init().",
        ),
        returns="IGV browser.",
    )
    def igv(
        self,
        region: base_params.region,
        tracks: Optional[List] = None,
        init: bool = True,
    ) -> igv_notebook.Browser:
        # Parse region.
        region_prepped: Region = _parse_single_region(self, region)
        del region

        # Create config.
        config = self._igv_config(
            region=region_prepped,
            tracks=tracks,
        )

        # Initialise IGV notebook.
        if init:  # pragma: no cover
            igv_notebook.init()

        # Create IGV browser.
        browser = igv_notebook.Browser(config)

        return browser

    @_check_types
    @doc(
        summary="""
            Launch IGV and view sequence read alignments and SNP genotypes from
            the given sample.
        """,
        parameters=dict(
            sample="Sample identifier.",
            visibility_window="""
                Zoom level in base pairs at which alignment and SNP data will become
                visible.
            """,
            init="If True, call igv_notebook.init().",
            local_coverage_fallback="""
                If True, also add a coverage track computed locally from
                CNV HMM or SNP calls data. Unlike the alignments track,
                this does not depend on the BAM file remaining available
                at its hosted URL, so it provides a fallback source of
                coverage information if that URL stops working. It is
                not a substitute for the alignments track, since it
                contains only summarised coverage values, not individual
                reads.
            """,
            coverage_source="""
                Which data to compute the local coverage track from, if
                `local_coverage_fallback` is True. "cnv_hmm" gives
                windowed coverage genome-wide; "snp_calls" gives
                per-base read depth, but only at called SNP sites.
            """,
            coverage_track_path="""
                Path to write the local coverage track file to, if
                `local_coverage_fallback` is True. If not provided,
                defaults to a location under `results_cache` (which must
                therefore be configured).
            """,
        ),
    )
    def view_alignments(
        self,
        region: base_params.region,
        sample: str,
        visibility_window: int = 20_000,
        init: bool = True,
        local_coverage_fallback: bool = False,
        coverage_source: str = "cnv_hmm",
        coverage_track_path: Optional[str] = None,
    ):
        # Parse region.
        region_prepped: Region = _parse_single_region(self, region)
        del region

        # Create tracks.
        tracks = self._igv_view_alignments_tracks(
            region=region_prepped,
            sample=sample,
            visibility_window=visibility_window,
            local_coverage_fallback=local_coverage_fallback,
            coverage_source=coverage_source,
            coverage_track_path=coverage_track_path,
        )

        # Create IGV browser.
        self.igv(region=region_prepped, tracks=tracks, init=init)

    @_check_types
    @doc(
        summary="""
            Launch IGV and view sequence read coverage from the given
            sample, computed locally from CNV HMM or SNP calls data.
        """,
        extended_summary="""
            Unlike `view_alignments()`, this does not depend on an
            externally-hosted BAM file, or on the WGS data catalog:
            coverage is derived directly from data already available in
            the Zarr arrays. It shows only summarised coverage depth,
            not individual aligned reads, so it is a fallback source of
            information about read coverage rather than a substitute for
            viewing alignments.
        """,
        parameters=dict(
            sample="Sample identifier.",
            source="""
                Which data to compute the coverage track from.
                "cnv_hmm" gives windowed coverage genome-wide;
                "snp_calls" gives per-base read depth, but only at
                called SNP sites.
            """,
            field="""
                Which coverage field to display, if `source` is
                "cnv_hmm". "NormCov" is coverage normalised against a set
                of reference windows; "RawCov" is unnormalised read
                counts per window. Ignored if `source` is "snp_calls".
            """,
            visibility_window="""
                Zoom level in base pairs at which SNP data will become
                visible.
            """,
            init="If True, call igv_notebook.init().",
            output_path="""
                Path to write the local coverage track file to. If not
                provided, defaults to a location under `results_cache`
                (which must therefore be configured).
            """,
        ),
    )
    def view_coverage(
        self,
        region: base_params.region,
        sample: str,
        source: str = "cnv_hmm",
        field: str = "NormCov",
        visibility_window: int = 20_000,
        init: bool = True,
        output_path: Optional[str] = None,
    ):
        # Parse region.
        region_prepped: Region = _parse_single_region(self, region)
        del region

        # Set up site filters tracks, consistent with view_alignments().
        tracks = self._igv_site_filters_tracks(
            contig=region_prepped.contig,
            visibility_window=visibility_window,
        )

        # Add the locally-computed coverage track. N.B., this does not
        # touch the WGS data catalog or any externally-hosted files.
        tracks.append(
            self._igv_local_coverage_track(
                region=region_prepped,
                sample=sample,
                visibility_window=visibility_window,
                source=source,
                field=field,
                output_path=output_path,
            )
        )

        # Create IGV browser.
        self.igv(region=region_prepped, tracks=tracks, init=init)
