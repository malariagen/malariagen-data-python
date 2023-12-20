from typing import Tuple, Optional

import allel  # type: ignore
import numpy as np
import pandas as pd
from numpydoc_decorator import doc  # type: ignore

from .. import veff
from ..util import Region, check_types, pandas_apply
from .snp_data import AnophelesSnpData
from . import base_params, frq_params


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

        # Set up cache variables.
        self._cache_annotator = None

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

    def _snp_effect_annotator(self):
        """Set up variant effect annotator."""
        if self._cache_annotator is None:
            self._cache_annotator = veff.Annotator(
                genome=self.open_genome(), genome_features=self.genome_features()
            )
        return self._cache_annotator

    @check_types
    @doc(
        summary="Compute variant effects for a gene transcript.",
        returns="""
            A dataframe of all possible SNP variants and their effects, one row
            per variant.
        """,
    )
    def snp_effects(
        self,
        transcript: base_params.transcript,
        site_mask: Optional[base_params.site_mask] = None,
    ) -> pd.DataFrame:
        # Setup initial dataframe of SNPs.
        _, df_snps = self._snp_df_for_transcript(transcript=transcript)

        # Setup variant effect annotator.
        ann = self._snp_effect_annotator()

        # Apply mask if requested.
        if site_mask is not None:
            loc_sites = df_snps[f"pass_{site_mask}"]
            df_snps = df_snps.loc[loc_sites]

        # Reset index after filtering.
        df_snps.reset_index(inplace=True, drop=True)

        # Add effects to the dataframe.
        ann.get_effects(transcript=transcript, variants=df_snps)

        return df_snps

    @check_types
    @doc(
        summary="""
            Compute SNP allele frequencies for a gene transcript.
        """,
        returns="""
            A dataframe of SNP allele frequencies, one row per variant allele.
        """,
        notes="""
            Cohorts with fewer samples than `min_cohort_size` will be excluded from
            output data frame.
        """,
    )
    def snp_allele_frequencies(
        self,
        transcript: base_params.transcript,
        cohorts: base_params.cohorts,
        sample_query: Optional[base_params.sample_query] = None,
        min_cohort_size: base_params.min_cohort_size = 10,
        site_mask: Optional[base_params.site_mask] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        drop_invariant: frq_params.drop_invariant = True,
        effects: frq_params.effects = True,
    ) -> pd.DataFrame:
        # Access sample metadata.
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )

        # Set up initial dataframe of SNPs.
        region, df_snps = self._snp_df_for_transcript(transcript=transcript)

        # Get genotypes.
        gt = self.snp_genotypes(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            field="GT",
        )

        # Slice to feature location.
        with self._dask_progress(desc="Load SNP genotypes"):
            gt = gt.compute()

        # Build coh dict.
        coh_dict = self._locate_cohorts(cohorts=cohorts, df_samples=df_samples)

        # Count alleles.
        freq_cols = dict()
        cohorts_iterator = self._progress(
            coh_dict.items(), desc="Compute allele frequencies"
        )
        for coh, loc_coh in cohorts_iterator:
            n_samples = np.count_nonzero(loc_coh)
            if n_samples >= min_cohort_size:
                gt_coh = np.compress(loc_coh, gt, axis=1)
                ac_coh = allel.GenotypeArray(gt_coh).count_alleles(max_allele=3)
                af_coh = ac_coh.to_frequencies()
                freq_cols["frq_" + coh] = af_coh[:, 1:].flatten()

        # Build a dataframe with the frequency columns.
        df_freqs = pd.DataFrame(freq_cols)

        # Compute max_af.
        df_max_af = pd.DataFrame({"max_af": df_freqs.max(axis=1)})

        # Build the final dataframe.
        df_snps.reset_index(drop=True, inplace=True)
        df_snps = pd.concat([df_snps, df_freqs, df_max_af], axis=1)

        # Apply site mask if requested.
        if site_mask is not None:
            loc_sites = df_snps[f"pass_{site_mask}"]
            df_snps = df_snps.loc[loc_sites]

        # Drop invariants.
        if drop_invariant:
            loc_variant = df_snps["max_af"] > 0
            df_snps = df_snps.loc[loc_variant]

        # Reset index after filtering.
        df_snps.reset_index(inplace=True, drop=True)

        if effects:
            # Add effect annotations.
            ann = self._snp_effect_annotator()
            ann.get_effects(
                transcript=transcript, variants=df_snps, progress=self._progress
            )

            # Add label.
            df_snps["label"] = pandas_apply(
                _make_snp_label_effect,
                df_snps,
                columns=["contig", "position", "ref_allele", "alt_allele", "aa_change"],
            )

            # Set index.
            df_snps.set_index(
                ["contig", "position", "ref_allele", "alt_allele", "aa_change"],
                inplace=True,
            )

        else:
            # Add label.
            df_snps["label"] = pandas_apply(
                _make_snp_label,
                df_snps,
                columns=["contig", "position", "ref_allele", "alt_allele"],
            )

            # Set index.
            df_snps.set_index(
                ["contig", "position", "ref_allele", "alt_allele"],
                inplace=True,
            )

        # Add dataframe metadata.
        gene_name = self._transcript_to_gene_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        df_snps.attrs["title"] = title

        return df_snps

    # TODO: aa_allele_frequencies
    # TODO: snp_allele_frequencies_advanced
    # TODO: aa_allele_frequencies_advanced
    # TODO: plot_frequencies_heatmap
    # TODO: plot_frequencies_time_series
    # TODO: plot_frequencies_interactive_map


def _make_snp_label(contig, position, ref_allele, alt_allele):
    return f"{contig}:{position:,} {ref_allele}>{alt_allele}"


def _make_snp_label_effect(contig, position, ref_allele, alt_allele, aa_change):
    label = f"{contig}:{position:,} {ref_allele}>{alt_allele}"
    if isinstance(aa_change, str):
        label += f" ({aa_change})"
    return label
