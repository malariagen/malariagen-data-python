from functools import cached_property
from typing import Optional, Dict, Union, Callable, List
import warnings

import allel  # type: ignore
import numpy as np
import numpy.typing as npt
import pandas as pd
from numpydoc_decorator import doc  # type: ignore
import xarray as xr
import numba  # type: ignore

from .. import veff
from ..util import (
    _check_types,
    _pandas_apply,
)
from .safe_query import validate_query
from .snp_data import AnophelesSnpData
from .frq_base import (
    _prep_samples_for_cohort_grouping,
    _build_cohorts_from_sample_grouping,
    _add_frequency_ci,
)
from .sample_metadata import _locate_cohorts
from .frq_base import AnophelesFrequencyAnalysis
from . import base_params, frq_params


AA_CHANGE_QUERY = (
    "effect in ['NON_SYNONYMOUS_CODING', 'START_LOST', 'STOP_LOST', 'STOP_GAINED']"
)


class AnophelesSnpFrequencyAnalysis(AnophelesSnpData, AnophelesFrequencyAnalysis):
    def __init__(
        self,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

    def _snp_df_melt(self, *, ds_snp: xr.Dataset) -> pd.DataFrame:
        """Set up a dataframe with SNP site and filter data,
        melting each alternate allele into a separate row."""

        with self._spinner(desc="Prepare SNP dataframe"):
            contig_index = ds_snp["variant_contig"].values[0]
            contig = ds_snp.attrs["contigs"][contig_index]
            pos = ds_snp["variant_position"].values
            alleles = ds_snp["variant_allele"].values
            ref = alleles[:, 0]
            alt = alleles[:, 1:]

            filter_pass = dict()
            for m in self.site_mask_ids:
                x = ds_snp[f"variant_filter_pass_{m}"].values
                filter_pass[m] = x

            cols = {
                "contig": contig,
                "position": np.repeat(pos, 3),
                "ref_allele": np.repeat(ref.astype("U1"), 3),
                "alt_allele": alt.astype("U1").flatten(),
            }

            for m in self.site_mask_ids:
                x = filter_pass[m]
                cols[f"pass_{m}"] = np.repeat(x, 3)

            df_snps = pd.DataFrame(cols)

        return df_snps

    @cached_property
    def _snp_effect_annotator(self):
        """Set up variant effect annotator."""
        return veff.Annotator(
            genome=self.open_genome(), genome_features=self.genome_features()
        )

    @_check_types
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
        # Access SNP data.
        ds_snp = self.snp_variants(
            region=transcript,
            site_mask=site_mask,
        )

        df_snps = self._snp_df_melt(ds_snp=ds_snp)

        ann = self._snp_effect_annotator

        ann.get_effects(transcript=transcript, variants=df_snps)

        return df_snps

    @_check_types
    @doc(
        summary="""
            Compute SNP allele frequencies for a gene transcript or genomic region.
        """,
        returns="""
            A dataframe of SNP allele frequencies, one row per variant allele. The variant alleles are indexed by
            their contig, their position, the reference allele, the alternate allele and the associated amino acid change.
            The columns are split into three categories: there is one column for each taxon filter (e.g., pass_funestus, pass_gamb_colu, ...) containing whether the site of the variant allele passes the filter;
            there is then 1 column for each cohort containing the frequency of the variant allele within the cohort, additionally there is a column `max_af` containing the maximum allele frequency of the variant allele accross all cohorts;
            finally, there are 9 columns describing the variant allele: `transcript` contains the gene transcript used for this analysis (when provided),
            `effect` is the effect of the allele change,
            `impact`is the impact of the allele change,
            `ref_codon` is the reference codon,
            `alt_codon` is the altered codon with the variant allele,
            `aa_pos` is the position of the amino acid,
            `ref_aa` is the reference amino acid,
            `alt_aa` is the altered amino acid with the variant allele,
            and `label` is the label of the variant allele.
        """,
        notes="""
            Cohorts with fewer samples than `min_cohort_size` will be excluded from
            output data frame.
        """,
    )
    def snp_allele_frequencies(
        self,
        transcript: Optional[base_params.transcript] = None,
        region: Optional[base_params.region] = None,
        cohorts: Optional[base_params.cohorts] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        min_cohort_size: base_params.min_cohort_size = 10,
        site_mask: Optional[base_params.site_mask] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        drop_invariant: frq_params.drop_invariant = True,
        effects: frq_params.effects = True,
        include_counts: frq_params.include_counts = False,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> pd.DataFrame:
        if transcript is None and region is None:
            raise ValueError("Provide either transcript or region.")
        if transcript is not None and region is not None:
            raise ValueError("Provide only one of transcript or region, not both.")

        if region is None:
            region = transcript

        df_samples = self.sample_metadata(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
        )

        coh_dict = _locate_cohorts(
            cohorts=cohorts, data=df_samples, min_cohort_size=min_cohort_size
        )

        # Access SNP data.
        ds_snp = self.snp_calls(
            region=region,
            site_mask=site_mask,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            chunks=chunks,
            inline_array=inline_array,
        )

        if ds_snp.sizes["variants"] == 0:
            raise ValueError("No SNPs available for the given region and site mask.")

        gt = ds_snp["call_genotype"].data
        with self._dask_progress(desc="Load SNP genotypes"):
            gt = gt.compute()

        df_snps = self._snp_df_melt(ds_snp=ds_snp)

        count_cols = dict()
        nobs_cols = dict()
        freq_cols = dict()
        cohorts_iterator = self._progress(
            coh_dict.items(), desc="Compute allele frequencies"
        )
        for coh, loc_coh in cohorts_iterator:
            n_samples = np.count_nonzero(loc_coh)
            if n_samples < min_cohort_size:
                raise ValueError(
                    f"Not enough samples ({n_samples}) for minimum "
                    f"cohort size ({min_cohort_size})"
                )
            gt_coh = np.compress(loc_coh, gt, axis=1)
            ac_coh = np.asarray(allel.GenotypeArray(gt_coh).count_alleles(max_allele=3))
            an_coh = np.sum(ac_coh, axis=1)[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                af_coh = np.where(an_coh > 0, ac_coh / an_coh, np.nan)

            frq = af_coh[:, 1:].flatten()
            freq_cols["frq_" + coh] = frq
            count = ac_coh[:, 1:].flatten()
            count_cols["count_" + coh] = count
            nobs = np.repeat(an_coh[:, 0], 3)
            nobs_cols["nobs_" + coh] = nobs

        # Build a dataframe with the frequency columns.
        df_freqs = pd.DataFrame(freq_cols)
        df_counts = pd.DataFrame(count_cols)
        df_nobs = pd.DataFrame(nobs_cols)

        df_max_af = pd.DataFrame({"max_af": df_freqs.max(axis=1)})

        df_snps.reset_index(drop=True, inplace=True)
        if include_counts:
            df_snps = pd.concat(
                [df_snps, df_freqs, df_max_af, df_counts, df_nobs], axis=1
            )
        else:
            df_snps = pd.concat([df_snps, df_freqs, df_max_af], axis=1)

        if drop_invariant:
            loc_variant = df_snps["max_af"] > 0

            # Check for no SNPs remaining after dropping invariants.
            if np.count_nonzero(loc_variant) == 0:
                raise ValueError("No SNPs remaining after dropping invariant SNPs.")

            df_snps = df_snps.loc[loc_variant]

        # Reset index after filtering.
        df_snps.reset_index(inplace=True, drop=True)

        if effects and (transcript is not None):
            ann = self._snp_effect_annotator
            ann.get_effects(
                transcript=transcript, variants=df_snps, progress=self._progress
            )

            df_snps["label"] = _pandas_apply(
                _make_snp_label_effect,
                df_snps,
                columns=["contig", "position", "ref_allele", "alt_allele", "aa_change"],
            )

            df_snps.set_index(
                ["contig", "position", "ref_allele", "alt_allele", "aa_change"],
                inplace=True,
            )

        else:
            # No transcript (or effects=False): do not require effect annotation.
            df_snps["label"] = _pandas_apply(
                _make_snp_label,
                df_snps,
                columns=["contig", "position", "ref_allele", "alt_allele"],
            )

            df_snps.set_index(
                ["contig", "position", "ref_allele", "alt_allele"],
                inplace=True,
            )

        # Add dataframe metadata.
        if transcript is not None:
            gene_name = self._transcript_to_parent_name(transcript)
            title = transcript
            if gene_name:
                title += f" ({gene_name})"
        else:
            title = str(region)

        title += " SNP frequencies"
        df_snps.attrs["title"] = title

        return df_snps

    @_check_types
    @doc(
        summary="""
            Compute amino acid substitution frequencies for a gene transcript.
        """,
        returns="""
            A dataframe of amino acid allele frequencies, one row per variant. The variants are indexed by
            their amino acid change, their contig, their position.
            The columns are split into two categories: there is 1 column for each cohort containing the frequency of the amino acid change within the cohort, additionally there is a column `max_af` containing the maximum frequency of the amino acid change across all cohorts;
            finally, there are 9 columns describing the variant allele: `transcript` contains the gene transcript used for this analysis,
            `effect` is the effect of the allele change,
            `impact`is the impact of the allele change,
            `ref_allele` is the reference allele,
            `alt_allele` is the alternate allele,
            `aa_pos` is the position of the amino acid,
            `ref_aa` is the reference amino acid,
            `alt_aa` is the altered amino acid with the variant allele,
            and `label` is the label of the variant allele.
        """,
        notes="""
            Cohorts with fewer samples than `min_cohort_size` will be excluded from
            output data frame.
        """,
    )
    def aa_allele_frequencies(
        self,
        transcript: base_params.transcript,
        cohorts: base_params.cohorts,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = 10,
        site_mask: Optional[base_params.site_mask] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        drop_invariant: frq_params.drop_invariant = True,
        include_counts: frq_params.include_counts = False,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> pd.DataFrame:
        df_snps = self.snp_allele_frequencies(
            transcript=transcript,
            cohorts=cohorts,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            min_cohort_size=min_cohort_size,
            site_mask=site_mask,
            sample_sets=sample_sets,
            drop_invariant=drop_invariant,
            effects=True,
            include_counts=include_counts,
            chunks=chunks,
            inline_array=inline_array,
        )
        df_snps.reset_index(inplace=True)

        # We just want aa change.
        df_ns_snps = df_snps.query(AA_CHANGE_QUERY).copy()

        # Handle case where no amino acid change SNPs are found.
        # This can legitimately happen for some transcript/site_mask/query
        # combinations. Return a well-formed empty DataFrame rather than raising,
        # to avoid transient test failures and to allow downstream code to handle
        # the empty result.
        if len(df_ns_snps) == 0:
            warnings.warn(
                "No amino acid change SNPs found for the given transcript "
                "and site mask. Returning an empty DataFrame.",
                stacklevel=2,
            )
            # Build an empty DataFrame with the expected schema.
            freq_cols = [col for col in df_snps.columns if col.startswith("frq_")]
            count_cols = [col for col in df_snps.columns if col.startswith("count_")]
            nobs_cols = [col for col in df_snps.columns if col.startswith("nobs_")]
            keep_cols = [
                "contig",
                "transcript",
                "aa_pos",
                "ref_allele",
                "ref_aa",
                "alt_aa",
                "effect",
                "impact",
                "grantham_score",
                "sneath_score",
            ]
            all_cols = (
                ["aa_change"]
                + freq_cols
                + ["max_af"]
                + keep_cols
                + ["alt_allele", "label", "position"]
            )
            if include_counts:
                all_cols = all_cols + count_cols + nobs_cols
            df_empty = pd.DataFrame(columns=all_cols)
            df_empty.set_index(["aa_change", "contig", "position"], inplace=True)

            # Add metadata.
            gene_name = self._transcript_to_parent_name(transcript)
            title = transcript
            if gene_name:
                title += f" ({gene_name})"
            title += " SNP frequencies"
            df_empty.attrs["title"] = title

            return df_empty

        # We need to worry about the possibility of the
        # same aa change due to SNPs at different positions. We cannot
        # sum frequencies of SNPs at different genomic positions. This
        # is why we group by position and aa_change, not just aa_change.

        # Group and sum to collapse multi variant allele changes.
        freq_cols = [col for col in df_ns_snps if col.startswith("frq_")]

        # Special handling here to ensure nans don't get summed to zero.

        def np_sum(g):
            return np.sum(g.values)

        agg: Dict[str, Union[Callable, str]] = {c: np_sum for c in freq_cols}

        # Add in counts and observations data if requested.
        if include_counts:
            count_cols = [col for col in df_ns_snps if col.startswith("count_")]
            for c in count_cols:
                agg[c] = "sum"
            nobs_cols = [col for col in df_ns_snps if col.startswith("nobs_")]
            for c in nobs_cols:
                agg[c] = "first"

        keep_cols = [
            "contig",
            "transcript",
            "aa_pos",
            "ref_allele",
            "ref_aa",
            "alt_aa",
            "effect",
            "impact",
            "grantham_score",
            "sneath_score",
        ]
        for c in keep_cols:
            agg[c] = "first"
        agg["alt_allele"] = lambda v: "{" + ",".join(v) + "}" if len(v) > 1 else v
        df_aaf = df_ns_snps.groupby(["position", "aa_change"]).agg(agg).reset_index()

        df_aaf["max_af"] = df_aaf[freq_cols].max(axis=1)

        df_aaf["label"] = _pandas_apply(
            _make_snp_label_aa,
            df_aaf,
            columns=["aa_change", "contig", "position", "ref_allele", "alt_allele"],
        )

        df_aaf = df_aaf.sort_values(["position", "aa_change"])

        df_aaf.set_index(["aa_change", "contig", "position"], inplace=True)

        gene_name = self._transcript_to_parent_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        df_aaf.attrs["title"] = title

        return df_aaf

    @_check_types
    @doc(
        summary="""
            Group samples by taxon, area (space) and period (time), then compute
            SNP allele frequencies.
        """,
        returns="""
            The resulting dataset contains data has dimensions "cohorts" and
            "variants". Variables prefixed with "cohort" are 1-dimensional
            arrays with data about the cohorts, such as the area, period, taxon
            and cohort size. Variables prefixed with "variant" are
            1-dimensional arrays with data about the variants, such as the
            contig, position, reference and alternate alleles. Variables
            prefixed with "event" are 2-dimensional arrays with the allele
            counts and frequency calculations.
        """,
    )
    def snp_allele_frequencies_advanced(
        self,
        transcript: base_params.transcript,
        area_by: frq_params.area_by,
        period_by: frq_params.period_by,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        min_cohort_size: base_params.min_cohort_size = 10,
        drop_invariant: frq_params.drop_invariant = True,
        variant_query: Optional[frq_params.variant_query] = None,
        site_mask: Optional[base_params.site_mask] = None,
        nobs_mode: frq_params.nobs_mode = frq_params.nobs_mode_default,
        ci_method: Optional[frq_params.ci_method] = frq_params.ci_method_default,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        taxon_by: frq_params.taxon_by = frq_params.taxon_by_default,
        filter_unassigned: Optional[frq_params.filter_unassigned] = None,
    ) -> xr.Dataset:
        df_samples = self.sample_metadata(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
        )

        df_samples = _prep_samples_for_cohort_grouping(
            df_samples=df_samples,
            area_by=area_by,
            period_by=period_by,
            taxon_by=taxon_by,
            filter_unassigned=filter_unassigned,
        )

        group_samples_by_cohort = df_samples.groupby([taxon_by, "area", "period"])

        df_cohorts = _build_cohorts_from_sample_grouping(
            group_samples_by_cohort=group_samples_by_cohort,
            min_cohort_size=min_cohort_size,
            taxon_by=taxon_by,
        )

        if len(df_cohorts) == 0:
            raise ValueError(
                "No cohorts available for the given sample selection parameters and minimum cohort size."
            )

        ds_snps = self.snp_calls(
            region=transcript,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            site_mask=site_mask,
            chunks=chunks,
            inline_array=inline_array,
        )

        if ds_snps.sizes["variants"] == 0:
            raise ValueError("No SNPs available for the given region and site mask.")

        gt = ds_snps["call_genotype"].data
        with self._dask_progress(desc="Load SNP genotypes"):
            gt = gt.compute()

        contigs = ds_snps.attrs["contigs"]
        variant_contig = np.repeat(
            [contigs[i] for i in ds_snps["variant_contig"].values], 3
        )
        variant_position = np.repeat(ds_snps["variant_position"].values, 3)
        alleles = ds_snps["variant_allele"].values
        variant_ref_allele = np.repeat(alleles[:, 0], 3)
        variant_alt_allele = alleles[:, 1:].flatten()
        variant_pass = dict()
        for site_mask in self.site_mask_ids:
            variant_pass[site_mask] = np.repeat(
                ds_snps[f"variant_filter_pass_{site_mask}"].values, 3
            )

        n_variants, n_cohorts = len(variant_position), len(df_cohorts)
        count: npt.NDArray[np.float64] = np.zeros((n_variants, n_cohorts), dtype=int)
        nobs: npt.NDArray[np.float64] = np.zeros((n_variants, n_cohorts), dtype=int)

        cohorts_iterator = self._progress(
            enumerate(df_cohorts.itertuples()),
            total=len(df_cohorts),
            desc="Compute SNP allele frequencies",
        )
        for cohort_index, cohort in cohorts_iterator:
            cohort_taxon = getattr(cohort, taxon_by)
            cohort_key = cohort_taxon, cohort.area, cohort.period
            sample_indices = group_samples_by_cohort.indices[cohort_key]

            cohort_ac, cohort_an = _cohort_alt_allele_counts_melt(
                gt=gt,
                indices=sample_indices,
                max_allele=3,
            )
            count[:, cohort_index] = cohort_ac

            if nobs_mode == "called":
                nobs[:, cohort_index] = cohort_an
            else:
                if nobs_mode != "fixed":
                    raise RuntimeError(
                        f"Internal error: expected nobs_mode='fixed', "
                        f"got {nobs_mode!r}"
                    )
                nobs[:, cohort_index] = cohort.size * 2

        with np.errstate(divide="ignore", invalid="ignore"):
            frequency = count / nobs

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_af = np.nanmax(frequency, axis=1)

        df_variants_cols = {
            "contig": variant_contig,
            "position": variant_position,
            "ref_allele": variant_ref_allele.astype("U1"),
            "alt_allele": variant_alt_allele.astype("U1"),
            "max_af": max_af,
        }
        for site_mask in self.site_mask_ids:
            df_variants_cols[f"pass_{site_mask}"] = variant_pass[site_mask]
        df_variants = pd.DataFrame(df_variants_cols)

        if drop_invariant:
            loc_variant = max_af > 0

            if np.count_nonzero(loc_variant) == 0:
                raise ValueError("No SNPs remaining after dropping invariant SNPs.")

            df_variants = df_variants.loc[loc_variant].reset_index(drop=True)
            count = np.compress(loc_variant, count, axis=0)
            nobs = np.compress(loc_variant, nobs, axis=0)
            frequency = np.compress(loc_variant, frequency, axis=0)

        ann = self._snp_effect_annotator

        ann.get_effects(
            transcript=transcript, variants=df_variants, progress=self._progress
        )

        df_variants["label"] = _pandas_apply(
            _make_snp_label_effect,
            df_variants,
            columns=["contig", "position", "ref_allele", "alt_allele", "aa_change"],
        )

        ds_out = xr.Dataset()

        for coh_col in df_cohorts.columns:
            if coh_col == taxon_by:
                ds_out["cohort_taxon"] = "cohorts", df_cohorts[coh_col]
            else:
                ds_out[f"cohort_{coh_col}"] = "cohorts", df_cohorts[coh_col]

        for snp_col in df_variants.columns:
            ds_out[f"variant_{snp_col}"] = "variants", df_variants[snp_col]

        ds_out["event_count"] = ("variants", "cohorts"), count
        ds_out["event_nobs"] = ("variants", "cohorts"), nobs
        ds_out["event_frequency"] = ("variants", "cohorts"), frequency

        if variant_query is not None:
            validate_query(variant_query)
            loc_variants = np.asarray(df_variants.eval(variant_query))

            if np.count_nonzero(loc_variants) == 0:
                warnings.warn(
                    f"No SNPs remaining after applying variant query {variant_query!r}. "
                    "Returning dataset with zero variants.",
                    stacklevel=2,
                )

            variant_indices = np.where(loc_variants)[0]
            ds_out = ds_out.isel(variants=variant_indices)

        _add_frequency_ci(ds=ds_out, ci_method=ci_method)

        sorted_vars: List[str] = sorted([str(k) for k in ds_out.keys()])
        ds_out = ds_out[sorted_vars]

        gene_name = self._transcript_to_parent_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        ds_out.attrs["title"] = title

        return ds_out

    @_check_types
    @doc(
        summary="""
            Group samples by taxon, area (space) and period (time), then compute
            amino acid change allele frequencies.
        """,
        returns="""
            The resulting dataset contains data has dimensions "cohorts" and
            "variants". Variables prefixed with "cohort" are 1-dimensional
            arrays with data about the cohorts, such as the area, period, taxon
            and cohort size. Variables prefixed with "variant" are
            1-dimensional arrays with data about the variants, such as the
            contig, position, reference and alternate alleles. Variables
            prefixed with "event" are 2-dimensional arrays with the allele
            counts and frequency calculations.
        """,
    )
    def aa_allele_frequencies_advanced(
        self,
        transcript: base_params.transcript,
        area_by: frq_params.area_by,
        period_by: frq_params.period_by,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        min_cohort_size: base_params.min_cohort_size = 10,
        variant_query: Optional[frq_params.variant_query] = None,
        site_mask: Optional[base_params.site_mask] = None,
        nobs_mode: frq_params.nobs_mode = "called",
        ci_method: Optional[frq_params.ci_method] = "wilson",
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
        taxon_by: frq_params.taxon_by = frq_params.taxon_by_default,
        filter_unassigned: Optional[frq_params.filter_unassigned] = None,
    ) -> xr.Dataset:

        ds_snp_frq = self.snp_allele_frequencies_advanced(
            transcript=transcript,
            area_by=area_by,
            period_by=period_by,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            min_cohort_size=min_cohort_size,
            drop_invariant=True,
            variant_query=AA_CHANGE_QUERY,
            site_mask=site_mask,
            nobs_mode=nobs_mode,
            ci_method=None,
            chunks=chunks,
            inline_array=inline_array,
            taxon_by=taxon_by,
            filter_unassigned=filter_unassigned,
        )

        # We need to worry about the possibility of the
        # same aa change due to SNPs at different positions. We cannot
        # sum frequencies of SNPs at different genomic positions. This
        # is why we group by position and aa_change, not just aa_change.

        # Add in a special grouping column to work around the fact that xarray currently
        # doesn't support grouping by multiple variables in the same dimension.
        df_grouper = ds_snp_frq[
            ["variant_position", "variant_aa_change"]
        ].to_dataframe()
        grouper_var = df_grouper.apply(
            lambda row: "_".join([str(v) for v in row]), axis="columns"
        )
        ds_snp_frq["variant_position_aa_change"] = "variants", grouper_var

        group_by_aa_change = ds_snp_frq.groupby("variant_position_aa_change")

        ds_aa_frq = group_by_aa_change.map(_map_snp_to_aa_change_frq_ds)

        cohort_vars = [v for v in ds_snp_frq if v.startswith("cohort_")]
        for v in cohort_vars:
            ds_aa_frq[v] = ds_snp_frq[v]

        ds_aa_frq = ds_aa_frq.sortby(["variant_position", "variant_aa_change"])

        count = ds_aa_frq["event_count"].values
        nobs = ds_aa_frq["event_nobs"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            frequency = count / nobs
        ds_aa_frq["event_frequency"] = ("variants", "cohorts"), frequency

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_af = np.nanmax(ds_aa_frq["event_frequency"].values, axis=1)
        ds_aa_frq["variant_max_af"] = "variants", max_af

        variant_cols = [v for v in ds_aa_frq if v.startswith("variant_")]
        df_variants = ds_aa_frq[variant_cols].to_dataframe()
        df_variants.columns = [c.split("variant_")[1] for c in df_variants.columns]

        label = _pandas_apply(
            _make_snp_label_aa,
            df_variants,
            columns=["aa_change", "contig", "position", "ref_allele", "alt_allele"],
        )
        ds_aa_frq["variant_label"] = "variants", label

        if variant_query is not None:
            validate_query(variant_query)
            loc_variants = df_variants.eval(variant_query).values

            if np.count_nonzero(loc_variants) == 0:
                warnings.warn(
                    f"No SNPs remaining after applying variant query {variant_query!r}. "
                    "Returning dataset with zero variants.",
                    stacklevel=2,
                )

            variant_indices = np.where(loc_variants)[0]
            ds_aa_frq = ds_aa_frq.isel(variants=variant_indices)

        _add_frequency_ci(ds=ds_aa_frq, ci_method=ci_method)

        ds_aa_frq = ds_aa_frq[sorted(ds_aa_frq)]

        gene_name = self._transcript_to_parent_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        ds_aa_frq.attrs["title"] = title

        return ds_aa_frq

    def snp_genotype_allele_counts(
        self,
        transcript: base_params.transcript,
        snp_query: Optional[base_params.snp_query] = AA_CHANGE_QUERY,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        site_mask: Optional[base_params.site_mask] = None,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> pd.DataFrame:
        ds_snp = self.snp_calls(
            region=transcript,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            sample_sets=sample_sets,
            site_mask=None,
            chunks=chunks,
            inline_array=inline_array,
        )

        if ds_snp.sizes["variants"] == 0:  # pragma: no cover
            raise ValueError("No SNPs available for the given region and site mask.")

        gt = ds_snp["call_genotype"].data
        with self._dask_progress(desc="Load SNP genotypes"):
            gt = allel.GenotypeArray(gt.compute())

        df_snps = self._snp_df_melt(ds_snp=ds_snp)

        gt_counts = gt.to_allele_counts()
        gt_counts_melt = _melt_gt_counts(gt_counts.values)

        df_counts = pd.DataFrame(
            gt_counts_melt, columns=["count_" + s for s in ds_snp["sample_id"].values]
        )
        df_snps = pd.concat([df_snps, df_counts], axis=1)

        ann = self._snp_effect_annotator
        ann.get_effects(
            transcript=transcript, variants=df_snps, progress=self._progress
        )

        df_snps["label"] = _pandas_apply(
            _make_snp_label_effect,
            df_snps,
            columns=["contig", "position", "ref_allele", "alt_allele", "aa_change"],
        )

        if site_mask is not None:
            loc_sites = df_snps[f"pass_{site_mask}"]
            df_snps = df_snps.loc[loc_sites]

        if snp_query is not None:
            validate_query(snp_query)
            df_snps = df_snps.query(snp_query)

        return df_snps

    @_check_types
    @doc(
        summary=""" 
            Compute a simple SNP feature matrix for machine learning workflows.
        """,
        returns="""
            A pandas DataFrame where rows are samples or cohorts and columns are:
            total_snp_count: Total number of SNPs
            nonsynonymous_snp_count: Number of nonsynonymous SNPs  
            mean_allele_frequency: Mean allele frequency across all SNPs
            
            In cohort mode (cohorts provided): one row per cohort; counts based on
            unique (contig, position) sites; mean_allele_frequency from each frq_ column.
            
            In sample mode (cohorts is None): one row per sample; per-sample counts 
            from snp_genotype_allele_counts; mean AF from snp_allele_frequencies with 
            cohorts={"all": "True"}.
        """,
        notes="""
            This function provides a lightweight interface for creating ML-ready feature
            matrices from SNP data. It internally uses existing public methods and does
            not perform any machine learning operations itself.
        """,
    )
    def snp_feature_matrix(
        self,
        transcript: Optional[base_params.transcript] = None,
        region: Optional[base_params.region] = None,
        cohorts: Optional[base_params.cohorts] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        min_cohort_size: base_params.min_cohort_size = 10,
        site_mask: Optional[base_params.site_mask] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> pd.DataFrame:
        if transcript is None and region is None:
            raise ValueError("Provide either transcript or region.")
        if transcript is not None and region is not None:
            raise ValueError("Provide only one of transcript or region, not both.")

        if region is None:
            region = transcript

        if cohorts is not None:
            return self._snp_feature_matrix_cohort_mode(
                transcript=transcript,
                region=region,
                cohorts=cohorts,
                sample_query=sample_query,
                sample_query_options=sample_query_options,
                min_cohort_size=min_cohort_size,
                site_mask=site_mask,
                sample_sets=sample_sets,
                chunks=chunks,
                inline_array=inline_array,
            )
        else:
            return self._snp_feature_matrix_sample_mode(
                transcript=transcript,
                region=region,
                sample_query=sample_query,
                sample_query_options=sample_query_options,
                site_mask=site_mask,
                sample_sets=sample_sets,
                chunks=chunks,
                inline_array=inline_array,
            )

    def _snp_feature_matrix_cohort_mode(
        self,
        transcript: Optional[base_params.transcript],
        region: Optional[base_params.region],
        cohorts: base_params.cohorts,
        sample_query: Optional[base_params.sample_query],
        sample_query_options: Optional[base_params.sample_query_options],
        min_cohort_size: base_params.min_cohort_size,
        site_mask: Optional[base_params.site_mask],
        sample_sets: Optional[base_params.sample_sets],
        chunks: base_params.chunks,
        inline_array: base_params.inline_array,
    ) -> pd.DataFrame:
        df_frq = self.snp_allele_frequencies(
            transcript=transcript,
            region=region,
            cohorts=cohorts,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            min_cohort_size=min_cohort_size,
            site_mask=site_mask,
            sample_sets=sample_sets,
            chunks=chunks,
            inline_array=inline_array,
            drop_invariant=False,
        )

        df_frq_ns = self.snp_allele_frequencies(
            transcript=transcript,
            region=region,
            cohorts=cohorts,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            min_cohort_size=min_cohort_size,
            site_mask=site_mask,
            sample_sets=sample_sets,
            chunks=chunks,
            inline_array=inline_array,
            drop_invariant=False,
            effects=True,
        )

        freq_cols = [col for col in df_frq.columns if col.startswith("frq_")]

        total_snp_count = len(df_frq.groupby(["contig", "position"]))

        nonsyn_snps = df_frq_ns.query(AA_CHANGE_QUERY)
        nonsyn_snp_count = len(nonsyn_snps.groupby(["contig", "position"]))

        features = {}
        for cohort_col in freq_cols:
            cohort_name = cohort_col.replace("frq_", "")
            mean_af = df_frq[cohort_col].mean()

            features[cohort_name] = {
                "total_snp_count": total_snp_count,
                "nonsynonymous_snp_count": nonsyn_snp_count,
                "mean_allele_frequency": mean_af,
            }

        df_features = pd.DataFrame.from_dict(features, orient="index")

        if transcript is not None:
            gene_name = self._transcript_to_parent_name(transcript)
            title = transcript
            if gene_name:
                title += f" ({gene_name})"
        else:
            title = str(region)
        title += " SNP feature matrix"
        df_features.attrs["title"] = title

        return df_features

    def _snp_feature_matrix_sample_mode(
        self,
        transcript: Optional[base_params.transcript],
        region: Optional[base_params.region],
        sample_query: Optional[base_params.sample_query],
        sample_query_options: Optional[base_params.sample_query_options],
        site_mask: Optional[base_params.site_mask],
        sample_sets: Optional[base_params.sample_sets],
        chunks: base_params.chunks,
        inline_array: base_params.inline_array,
    ) -> pd.DataFrame:
        df_counts = self.snp_genotype_allele_counts(
            transcript=transcript,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            site_mask=site_mask,
            sample_sets=sample_sets,
            chunks=chunks,
            inline_array=inline_array,
            snp_query=None,
        )

        df_counts_ns = self.snp_genotype_allele_counts(
            transcript=transcript,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            site_mask=site_mask,
            sample_sets=sample_sets,
            chunks=chunks,
            inline_array=inline_array,
            snp_query=AA_CHANGE_QUERY,
        )

        df_frq = self.snp_allele_frequencies(
            transcript=transcript,
            region=region,
            cohorts={"all": "True"},
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            site_mask=site_mask,
            sample_sets=sample_sets,
            chunks=chunks,
            inline_array=inline_array,
            drop_invariant=False,
        )

        count_cols = [col for col in df_counts.columns if col.startswith("count_")]
        count_cols_ns = [
            col for col in df_counts_ns.columns if col.startswith("count_")
        ]

        mean_af = df_frq["frq_all"].mean()

        features = {}
        for sample_col in count_cols:
            sample_id = sample_col.replace("count_", "")

            sample_counts = df_counts[sample_col].values
            sample_snp_count = np.sum(sample_counts > 0)

            if sample_col in count_cols_ns:
                sample_counts_ns = df_counts_ns[sample_col].values
                sample_nonsyn_count = np.sum(sample_counts_ns > 0)
            else:
                sample_nonsyn_count = 0

            features[sample_id] = {
                "total_snp_count": sample_snp_count,
                "nonsynonymous_snp_count": sample_nonsyn_count,
                "mean_allele_frequency": mean_af,
            }

        df_features = pd.DataFrame.from_dict(features, orient="index")

        if transcript is not None:
            gene_name = self._transcript_to_parent_name(transcript)
            title = transcript
            if gene_name:
                title += f" ({gene_name})"
        else:
            title = str(region)
        title += " SNP feature matrix"
        df_features.attrs["title"] = title

        return df_features


@numba.jit(nopython=True)
def _melt_gt_counts(gt_counts):
    n_snps, n_samples, n_alleles = gt_counts.shape
    melted_counts = np.zeros((n_snps * (n_alleles - 1), n_samples), dtype=np.int32)

    for i in range(n_snps):
        for j in range(n_samples):
            for k in range(n_alleles - 1):
                melted_counts[(i * 3) + k][j] = gt_counts[i][j][k + 1]

    return melted_counts


def _make_snp_label(contig, position, ref_allele, alt_allele):
    return f"{contig}:{position:,} {ref_allele}>{alt_allele}"


def _make_snp_label_effect(contig, position, ref_allele, alt_allele, aa_change):
    label = f"{contig}:{position:,} {ref_allele}>{alt_allele}"
    if isinstance(aa_change, str):
        label += f" ({aa_change})"
    return label


def _make_snp_label_aa(aa_change, contig, position, ref_allele, alt_allele):
    label = f"{aa_change} ({contig}:{position:,} {ref_allele}>{alt_allele})"
    return label


def _cohort_alt_allele_counts_melt(*, gt, indices, max_allele):
    ac_alt_melt, an = _cohort_alt_allele_counts_melt_kernel(gt, indices, max_allele)
    an_melt = np.repeat(an, max_allele, axis=0)
    return ac_alt_melt, an_melt


@numba.njit
def _cohort_alt_allele_counts_melt_kernel(
    gt, sample_indices, max_allele
):  # pragma: no cover
    n_variants = gt.shape[0]
    n_samples = sample_indices.shape[0]
    ploidy = gt.shape[2]

    ac_alt_melt = np.zeros(n_variants * max_allele, dtype=np.int64)
    an = np.zeros(n_variants, dtype=np.int64)

    for i in range(n_variants):
        out_i_offset = (i * max_allele) - 1
        for j in range(n_samples):
            sample_index = sample_indices[j]
            for k in range(ploidy):
                allele = gt[i, sample_index, k]
                if allele > 0:
                    out_i = out_i_offset + allele
                    ac_alt_melt[out_i] += 1
                    an[i] += 1
                elif allele == 0:
                    an[i] += 1

    return ac_alt_melt, an


def _map_snp_to_aa_change_frq_ds(ds):
    # Keep only variables that make sense for amino acid substitutions.
    keep_vars = [
        "variant_contig",
        "variant_position",
        "variant_transcript",
        "variant_effect",
        "variant_impact",
        "variant_aa_pos",
        "variant_aa_change",
        "variant_ref_allele",
        "variant_ref_aa",
        "variant_alt_aa",
        "event_nobs",
    ]

    if ds.sizes["variants"] == 1:
        ds_out = ds[keep_vars + ["variant_alt_allele", "event_count"]]

    else:
        ds_out = ds[keep_vars].isel(variants=[0])

        count = ds["event_count"].values.sum(axis=0, keepdims=True)
        ds_out["event_count"] = ("variants", "cohorts"), count

        alt_allele = "{" + ",".join(ds["variant_alt_allele"].values) + "}"
        ds_out["variant_alt_allele"] = (
            "variants",
            np.array([alt_allele], dtype=object),
        )

    return ds_out
