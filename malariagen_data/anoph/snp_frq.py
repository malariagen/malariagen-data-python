from typing import Tuple

import numpy as np
import pandas as pd

from ..util import Region
from .snp_data import AnophelesSnpData


class AnophelesSnpFrequencyAnalysis(
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

    def _snp_df_for_transcript(self, *, transcript: str) -> Tuple[Region, pd.DataFrame]:
        """Set up a dataframe with SNP site and filter data for SNPs
        within a given gene transcript."""

        # Get feature direct from genome_features.
        gs = self.genome_features()

        with self._spinner(desc="Prepare SNP dataframe"):
            feature = gs[gs["ID"] == transcript].squeeze()
            if feature.empty:
                raise ValueError(
                    f"No genome feature ID found matching transcript {transcript}"
                )
            contig = feature.contig
            region = Region(contig, feature.start, feature.end)

            # Grab pos, ref and alt for chrom arm from snp_sites.
            ds_snp = self.snp_variants(region=region)
            pos = ds_snp["variant_position"].values
            alleles = ds_snp["variant_allele"].values
            ref = alleles[:, 0]
            alt = alleles[:, 1:]

            # Access site filters.
            filter_pass = dict()
            for m in self.site_mask_ids:
                x = ds_snp[f"variant_filter_pass_{m}"].values
                filter_pass[m] = x

            # Set up columns with contig, pos, ref, alt columns, melting
            # the data out to one row per alternate allele.
            cols = {
                "contig": contig,
                "position": np.repeat(pos, 3),
                "ref_allele": np.repeat(ref.astype("U1"), 3),
                "alt_allele": alt.astype("U1").flatten(),
            }

            # Add mask columns.
            for m in self.site_mask_ids:
                x = filter_pass[m]
                cols[f"pass_{m}"] = np.repeat(x, 3)

            # Construct dataframe.
            df_snps = pd.DataFrame(cols)

        return region, df_snps
