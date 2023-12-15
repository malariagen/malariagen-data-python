from typing import Dict, List, Optional

import igv_notebook
from numpydoc_decorator import doc

from ..util import Region, check_types, parse_single_region
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

    @check_types
    @doc(
        summary="Create an IGV browser and inject into the current notebook.",
        parameters=dict(
            tracks="Configuration for any additional tracks.",
        ),
        returns="IGV browser.",
    )
    def igv(
        self, region: base_params.region, tracks: Optional[List] = None
    ) -> igv_notebook.Browser:
        # Resolve region.
        region_prepped: Region = parse_single_region(self, region)
        del region

        # Create IGV config.
        config = {
            "reference": {
                "id": self._genome_ref_id,
                "name": self._genome_ref_name,
                "fastaURL": f"{self._url}{self._genome_fasta_path}",
                "indexURL": f"{self._url}{self._genome_fai_path}",
                "tracks": [
                    {
                        "name": "Genes",
                        "type": "annotation",
                        "format": "gff3",
                        "url": f"{self._url}{self._geneset_gff3_path}",
                        "indexed": False,
                    }
                ],
            },
            "locus": str(region_prepped),
        }
        if tracks:
            config["tracks"] = tracks

        igv_notebook.init()
        browser = igv_notebook.Browser(config)

        return browser

    def _view_alignments_add_site_filters_tracks(
        self, *, contig, visibility_window, tracks
    ):
        for site_mask in self.site_mask_ids:
            site_filters_vcf_url = f"{self._url}{self._major_version_path}/site_filters/{self._site_filters_analysis}/vcf/{site_mask}/{contig}_sitefilters.vcf.gz"  # noqa
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

    @check_types
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
        ),
    )
    def view_alignments(
        self,
        region: base_params.region,
        sample: str,
        visibility_window: int = 20_000,
    ):
        # Look up sample set for sample.
        sample_rec = self.sample_metadata().set_index("sample_id").loc[sample]
        sample_set = sample_rec["sample_set"]

        # Load data catalog.
        df_cat = self.wgs_data_catalog(sample_set=sample_set)

        # Locate record for sample.
        cat_rec = df_cat.set_index("sample_id").loc[sample]
        bam_url = cat_rec["alignments_bam"]
        vcf_url = cat_rec["snp_genotypes_vcf"]

        # Parse region.
        resolved_region: Region = parse_single_region(self, region)
        del region
        contig = resolved_region.contig

        # begin creating tracks
        tracks: List[Dict] = []

        # https://github.com/igvteam/igv-notebook/issues/3 -- resolved now
        # Set up site filters tracks.
        self._view_alignments_add_site_filters_tracks(
            contig=contig, visibility_window=visibility_window, tracks=tracks
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

        # Create IGV browser.
        self.igv(region=resolved_region, tracks=tracks)
