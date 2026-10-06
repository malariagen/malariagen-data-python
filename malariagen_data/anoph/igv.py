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

    def _igv_view_alignments_tracks(
        self,
        region: Region,
        sample: str,
        visibility_window: int = 20_000,
        snp_vcf_url: Optional[str] = None,
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

        # Set up site filters tracks.
        contig = region.contig
        tracks = self._igv_site_filters_tracks(
            contig=contig,
            visibility_window=visibility_window,
        )

        # Add SNPs track.
        snp_track = {
            "name": "SNPs",
            "format": "vcf",
            "type": "variant",
            "visibilityWindow": visibility_window,  # bp
            "height": 50,
        }
        if snp_vcf_url is not None:
            # A local VCF, e.g. from snp_calls_to_vcf(), has no index, so
            # IGV loads it in full.
            snp_track["url"] = snp_vcf_url
            snp_track["indexed"] = False
        else:
            vcf_url = cat_rec["snp_genotypes_vcf"]
            snp_track["url"] = vcf_url
            snp_track["indexURL"] = f"{vcf_url}.tbi"
        tracks.append(snp_track)

        # Add alignments track.
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
            snp_vcf_url="""
                Optional URL for a local VCF, e.g. one written by
                `snp_calls_to_vcf()`. The VCF is loaded without an index, so it
                should be small (e.g. a single sample over a region). For Jupyter
                Notebook and Lab, the VCF must be under the Jupyter startup
                directory tree; pass a URL relative to that tree (e.g.
                `/data/sample.vcf.gz`). If omitted, use the catalog VCF.
            """,
        ),
    )
    def view_alignments(
        self,
        region: base_params.region,
        sample: str,
        visibility_window: int = 20_000,
        init: bool = True,
        snp_vcf_url: Optional[str] = None,
    ):
        # Parse region.
        region_prepped: Region = _parse_single_region(self, region)
        del region

        # Create tracks.
        tracks = self._igv_view_alignments_tracks(
            region=region_prepped,
            sample=sample,
            visibility_window=visibility_window,
            snp_vcf_url=snp_vcf_url,
        )

        # Create IGV browser.
        self.igv(region=region_prepped, tracks=tracks, init=init)
