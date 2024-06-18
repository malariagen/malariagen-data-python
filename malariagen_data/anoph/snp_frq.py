from typing import Optional, Dict, Union, Callable, List
import warnings
from textwrap import dedent

import allel  # type: ignore
import numpy as np
import pandas as pd
from numpydoc_decorator import doc  # type: ignore
import xarray as xr
import numba  # type: ignore
import plotly.express as px  # type: ignore

from .. import veff
from ..util import check_types, pandas_apply
from .snp_data import AnophelesSnpData
from .sample_metadata import locate_cohorts
from . import base_params, frq_params, map_params, plotly_params


AA_CHANGE_QUERY = (
    "effect in ['NON_SYNONYMOUS_CODING', 'START_LOST', 'STOP_LOST', 'STOP_GAINED']"
)


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

    def _snp_df_melt(self, *, ds_snp: xr.Dataset) -> pd.DataFrame:
        """Set up a dataframe with SNP site and filter data,
        melting each alternate allele into a separate row."""

        with self._spinner(desc="Prepare SNP dataframe"):
            # Grab contig, pos, ref and alt.
            contig_index = ds_snp["variant_contig"].values[0]
            contig = ds_snp.attrs["contigs"][contig_index]
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

        return df_snps

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
        # Access SNP data.
        ds_snp = self.snp_variants(
            region=transcript,
            site_mask=site_mask,
        )

        # Setup initial dataframe of SNPs.
        df_snps = self._snp_df_melt(ds_snp=ds_snp)

        # Setup variant effect annotator.
        ann = self._snp_effect_annotator()

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
        include_counts: frq_params.include_counts = False,
    ) -> pd.DataFrame:
        # Access sample metadata.
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )

        # Build cohort dictionary, maps cohort labels to boolean indexers.
        coh_dict = locate_cohorts(cohorts=cohorts, data=df_samples)

        # Remove cohorts below minimum cohort size.
        coh_dict = {
            coh: loc_coh
            for coh, loc_coh in coh_dict.items()
            if np.count_nonzero(loc_coh) >= min_cohort_size
        }

        # Early check for no cohorts.
        if len(coh_dict) == 0:
            raise ValueError(
                "No cohorts available for the given sample selection parameters and minimum cohort size."
            )

        # Access SNP data.
        ds_snp = self.snp_calls(
            region=transcript,
            site_mask=site_mask,
            sample_sets=sample_sets,
            sample_query=sample_query,
        )

        # Early check for no SNPs.
        if ds_snp.sizes["variants"] == 0:  # pragma: no cover
            raise ValueError("No SNPs available for the given region and site mask.")

        # Access genotypes.
        gt = ds_snp["call_genotype"].data
        with self._dask_progress(desc="Load SNP genotypes"):
            gt = gt.compute()

        # Set up initial dataframe of SNPs.
        df_snps = self._snp_df_melt(ds_snp=ds_snp)

        # Count alleles.
        count_cols = dict()
        nobs_cols = dict()
        freq_cols = dict()
        cohorts_iterator = self._progress(
            coh_dict.items(), desc="Compute allele frequencies"
        )
        for coh, loc_coh in cohorts_iterator:
            n_samples = np.count_nonzero(loc_coh)
            assert n_samples >= min_cohort_size
            gt_coh = np.compress(loc_coh, gt, axis=1)
            ac_coh = np.asarray(allel.GenotypeArray(gt_coh).count_alleles(max_allele=3))
            an_coh = np.sum(ac_coh, axis=1)[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                af_coh = np.where(an_coh > 0, ac_coh / an_coh, np.nan)
            # Melt the frequencies so we get one row per alternate allele.
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

        # Compute max_af.
        df_max_af = pd.DataFrame({"max_af": df_freqs.max(axis=1)})

        # Build the final dataframe.
        df_snps.reset_index(drop=True, inplace=True)
        if include_counts:
            df_snps = pd.concat(
                [df_snps, df_freqs, df_max_af, df_counts, df_nobs], axis=1
            )
        else:
            df_snps = pd.concat([df_snps, df_freqs, df_max_af], axis=1)

        # Drop invariants.
        if drop_invariant:
            loc_variant = df_snps["max_af"] > 0

            # Check for no SNPs remaining after dropping invariants.
            if np.count_nonzero(loc_variant) == 0:  # pragma: no cover
                raise ValueError("No SNPs remaining after dropping invariant SNPs.")

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
        gene_name = self._transcript_to_parent_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        df_snps.attrs["title"] = title

        return df_snps

    @check_types
    @doc(
        summary="""
            Compute amino acid substitution frequencies for a gene transcript.
        """,
        returns="""
            A dataframe of amino acid allele frequencies, one row per
            substitution.
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
        min_cohort_size: Optional[base_params.min_cohort_size] = 10,
        site_mask: Optional[base_params.site_mask] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        drop_invariant: frq_params.drop_invariant = True,
        include_counts: frq_params.include_counts = False,
    ) -> pd.DataFrame:
        df_snps = self.snp_allele_frequencies(
            transcript=transcript,
            cohorts=cohorts,
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
            site_mask=site_mask,
            sample_sets=sample_sets,
            drop_invariant=drop_invariant,
            effects=True,
            include_counts=include_counts,
        )
        df_snps.reset_index(inplace=True)

        # We just want aa change.
        df_ns_snps = df_snps.query(AA_CHANGE_QUERY).copy()

        # Early check for no matching SNPs.
        if len(df_ns_snps) == 0:  # pragma: no cover
            raise ValueError(
                "No amino acid change SNPs found for the given transcript and site mask."
            )

        # N.B., we need to worry about the possibility of the
        # same aa change due to SNPs at different positions. We cannot
        # sum frequencies of SNPs at different genomic positions. This
        # is why we group by position and aa_change, not just aa_change.

        # Group and sum to collapse multi variant allele changes.
        freq_cols = [col for col in df_ns_snps if col.startswith("frq_")]

        # Special handling here to ensure nans don't get summed to zero.
        # See also https://github.com/pandas-dev/pandas/issues/20824#issuecomment-705376621
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

        keep_cols = (
            "contig",
            "transcript",
            "aa_pos",
            "ref_allele",
            "ref_aa",
            "alt_aa",
            "effect",
            "impact",
        )
        for c in keep_cols:
            agg[c] = "first"
        agg["alt_allele"] = lambda v: "{" + ",".join(v) + "}" if len(v) > 1 else v
        df_aaf = df_ns_snps.groupby(["position", "aa_change"]).agg(agg).reset_index()

        # Compute new max_af.
        df_aaf["max_af"] = df_aaf[freq_cols].max(axis=1)

        # Add label.
        df_aaf["label"] = pandas_apply(
            _make_snp_label_aa,
            df_aaf,
            columns=["aa_change", "contig", "position", "ref_allele", "alt_allele"],
        )

        # Sort by genomic position.
        df_aaf = df_aaf.sort_values(["position", "aa_change"])

        # Set index.
        df_aaf.set_index(["aa_change", "contig", "position"], inplace=True)

        # Add metadata.
        gene_name = self._transcript_to_parent_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        df_aaf.attrs["title"] = title

        return df_aaf

    @check_types
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
        min_cohort_size: base_params.min_cohort_size = 10,
        drop_invariant: frq_params.drop_invariant = True,
        variant_query: Optional[frq_params.variant_query] = None,
        site_mask: Optional[base_params.site_mask] = None,
        nobs_mode: frq_params.nobs_mode = frq_params.nobs_mode_default,
        ci_method: Optional[frq_params.ci_method] = frq_params.ci_method_default,
    ) -> xr.Dataset:
        # Load sample metadata.
        df_samples = self.sample_metadata(
            sample_sets=sample_sets, sample_query=sample_query
        )

        # Prepare sample metadata for cohort grouping.
        df_samples = _prep_samples_for_cohort_grouping(
            df_samples=df_samples,
            area_by=area_by,
            period_by=period_by,
        )

        # Group samples to make cohorts.
        group_samples_by_cohort = df_samples.groupby(["taxon", "area", "period"])

        # Build cohorts dataframe.
        df_cohorts = _build_cohorts_from_sample_grouping(
            group_samples_by_cohort=group_samples_by_cohort,
            min_cohort_size=min_cohort_size,
        )

        # Early check for no cohorts.
        if len(df_cohorts) == 0:
            raise ValueError(
                "No cohorts available for the given sample selection parameters and minimum cohort size."
            )

        # Access SNP calls.
        ds_snps = self.snp_calls(
            region=transcript,
            sample_sets=sample_sets,
            sample_query=sample_query,
            site_mask=site_mask,
        )

        # Early check for no SNPs.
        if ds_snps.sizes["variants"] == 0:  # pragma: no cover
            raise ValueError("No SNPs available for the given region and site mask.")

        # Access genotypes.
        gt = ds_snps["call_genotype"].data
        with self._dask_progress(desc="Load SNP genotypes"):
            gt = gt.compute()

        # Set up variant variables.
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

        # Set up main event variables.
        n_variants, n_cohorts = len(variant_position), len(df_cohorts)
        count = np.zeros((n_variants, n_cohorts), dtype=int)
        nobs = np.zeros((n_variants, n_cohorts), dtype=int)

        # Build event count and nobs for each cohort.
        cohorts_iterator = self._progress(
            enumerate(df_cohorts.itertuples()),
            total=len(df_cohorts),
            desc="Compute SNP allele frequencies",
        )
        for cohort_index, cohort in cohorts_iterator:
            cohort_key = cohort.taxon, cohort.area, cohort.period
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
                assert nobs_mode == "fixed"
                nobs[:, cohort_index] = cohort.size * 2

        # Compute frequency.
        with np.errstate(divide="ignore", invalid="ignore"):
            # Ignore division warnings.
            frequency = count / nobs

        # Compute maximum frequency over cohorts.
        with warnings.catch_warnings():
            # Ignore "All-NaN slice encountered" warnings.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_af = np.nanmax(frequency, axis=1)

        # Make dataframe of SNPs.
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

        # Deal with SNP alleles not observed.
        if drop_invariant:
            loc_variant = max_af > 0

            # Check for no SNPs remaining after dropping invariants.
            if np.count_nonzero(loc_variant) == 0:  # pragma: no cover
                raise ValueError("No SNPs remaining after dropping invariant SNPs.")

            df_variants = df_variants.loc[loc_variant].reset_index(drop=True)
            count = np.compress(loc_variant, count, axis=0)
            nobs = np.compress(loc_variant, nobs, axis=0)
            frequency = np.compress(loc_variant, frequency, axis=0)

        # Set up variant effect annotator.
        ann = self._snp_effect_annotator()

        # Add effects to the dataframe.
        ann.get_effects(
            transcript=transcript, variants=df_variants, progress=self._progress
        )

        # Add variant labels.
        df_variants["label"] = pandas_apply(
            _make_snp_label_effect,
            df_variants,
            columns=["contig", "position", "ref_allele", "alt_allele", "aa_change"],
        )

        # Build the output dataset.
        ds_out = xr.Dataset()

        # Cohort variables.
        for coh_col in df_cohorts.columns:
            ds_out[f"cohort_{coh_col}"] = "cohorts", df_cohorts[coh_col]

        # Variant variables.
        for snp_col in df_variants.columns:
            ds_out[f"variant_{snp_col}"] = "variants", df_variants[snp_col]

        # Event variables.
        ds_out["event_count"] = ("variants", "cohorts"), count
        ds_out["event_nobs"] = ("variants", "cohorts"), nobs
        ds_out["event_frequency"] = ("variants", "cohorts"), frequency

        # Apply variant query.
        if variant_query is not None:
            loc_variants = df_variants.eval(variant_query).values

            # Check for no SNPs remaining after applying variant query.
            if np.count_nonzero(loc_variants) == 0:
                raise ValueError(
                    f"No SNPs remaining after applying variant query {variant_query!r}."
                )

            ds_out = ds_out.isel(variants=loc_variants)

        # Add confidence intervals.
        _add_frequency_ci(ds=ds_out, ci_method=ci_method)

        # Tidy up display by sorting variables.
        sorted_vars: List[str] = sorted([str(k) for k in ds_out.keys()])
        ds_out = ds_out[sorted_vars]

        # Add metadata.
        gene_name = self._transcript_to_parent_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        ds_out.attrs["title"] = title

        return ds_out

    @check_types
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
        min_cohort_size: base_params.min_cohort_size = 10,
        variant_query: Optional[frq_params.variant_query] = None,
        site_mask: Optional[base_params.site_mask] = None,
        nobs_mode: frq_params.nobs_mode = "called",
        ci_method: Optional[frq_params.ci_method] = "wilson",
    ) -> xr.Dataset:
        # Begin by computing SNP allele frequencies.
        ds_snp_frq = self.snp_allele_frequencies_advanced(
            transcript=transcript,
            area_by=area_by,
            period_by=period_by,
            sample_sets=sample_sets,
            sample_query=sample_query,
            min_cohort_size=min_cohort_size,
            drop_invariant=True,  # always drop invariant for aa frequencies
            variant_query=AA_CHANGE_QUERY,  # we'll also apply a variant query later
            site_mask=site_mask,
            nobs_mode=nobs_mode,
            ci_method=None,  # we will recompute confidence intervals later
        )

        # N.B., we need to worry about the possibility of the
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

        # Group by position and amino acid change.
        group_by_aa_change = ds_snp_frq.groupby("variant_position_aa_change")

        # Apply aggregation.
        ds_aa_frq = group_by_aa_change.map(_map_snp_to_aa_change_frq_ds)

        # Add back in cohort variables, unaffected by aggregation.
        cohort_vars = [v for v in ds_snp_frq if v.startswith("cohort_")]
        for v in cohort_vars:
            ds_aa_frq[v] = ds_snp_frq[v]

        # Sort by genomic position.
        ds_aa_frq = ds_aa_frq.sortby(["variant_position", "variant_aa_change"])

        # Recompute frequency.
        count = ds_aa_frq["event_count"].values
        nobs = ds_aa_frq["event_nobs"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            frequency = count / nobs  # ignore division warnings
        ds_aa_frq["event_frequency"] = ("variants", "cohorts"), frequency

        # Recompute max frequency over cohorts.
        with warnings.catch_warnings():
            # Ignore "All-NaN slice encountered" warnings.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_af = np.nanmax(ds_aa_frq["event_frequency"].values, axis=1)
        ds_aa_frq["variant_max_af"] = "variants", max_af

        # Set up variant dataframe, useful intermediate.
        variant_cols = [v for v in ds_aa_frq if v.startswith("variant_")]
        df_variants = ds_aa_frq[variant_cols].to_dataframe()
        df_variants.columns = [c.split("variant_")[1] for c in df_variants.columns]

        # Assign new variant label.
        label = pandas_apply(
            _make_snp_label_aa,
            df_variants,
            columns=["aa_change", "contig", "position", "ref_allele", "alt_allele"],
        )
        ds_aa_frq["variant_label"] = "variants", label

        # Apply variant query if given.
        if variant_query is not None:
            loc_variants = df_variants.eval(variant_query).values

            # Check for no SNPs remaining after applying variant query.
            if np.count_nonzero(loc_variants) == 0:
                raise ValueError(
                    f"No SNPs remaining after applying variant query {variant_query!r}."
                )

            ds_aa_frq = ds_aa_frq.isel(variants=loc_variants)

        # Compute new confidence intervals.
        _add_frequency_ci(ds=ds_aa_frq, ci_method=ci_method)

        # Tidy up display by sorting variables.
        ds_aa_frq = ds_aa_frq[sorted(ds_aa_frq)]

        gene_name = self._transcript_to_parent_name(transcript)
        title = transcript
        if gene_name:
            title += f" ({gene_name})"
        title += " SNP frequencies"
        ds_aa_frq.attrs["title"] = title

        return ds_aa_frq

    @check_types
    @doc(
        summary="""
            Plot a heatmap from a pandas DataFrame of frequencies, e.g., output
            from `snp_allele_frequencies()` or `gene_cnv_frequencies()`.
        """,
        parameters=dict(
            df="""
                A DataFrame of frequencies, e.g., output from
                `snp_allele_frequencies()` or `gene_cnv_frequencies()`.
            """,
            index="""
                One or more column headers that are present in the input dataframe.
                This becomes the heatmap y-axis row labels. The column/s must
                produce a unique index.
            """,
            max_len="""
                Displaying large styled dataframes may cause ipython notebooks to
                crash. If the input dataframe is larger than this value, an error
                will be raised.
            """,
            col_width="""
                Plot width per column in pixels (px).
            """,
            row_height="""
                Plot height per row in pixels (px).
            """,
            kwargs="""
                Passed through to `px.imshow()`.
            """,
        ),
        notes="""
            It's recommended to filter the input DataFrame to just rows of interest,
            i.e., fewer rows than `max_len`.
        """,
    )
    def plot_frequencies_heatmap(
        self,
        df: pd.DataFrame,
        index: Optional[Union[str, List[str]]] = "label",
        max_len: Optional[int] = 100,
        col_width: int = 40,
        row_height: int = 20,
        x_label: plotly_params.x_label = "Cohorts",
        y_label: plotly_params.y_label = "Variants",
        colorbar: plotly_params.colorbar = True,
        width: plotly_params.fig_width = None,
        height: plotly_params.fig_height = None,
        text_auto: plotly_params.text_auto = ".0%",
        aspect: plotly_params.aspect = "auto",
        color_continuous_scale: plotly_params.color_continuous_scale = "Reds",
        title: plotly_params.title = True,
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        **kwargs,
    ) -> plotly_params.figure:
        # Check len of input.
        if max_len and len(df) > max_len:
            raise ValueError(
                dedent(
                    f"""
                Input DataFrame is longer than max_len parameter value {max_len}, which means
                that the plot is likely to be very large. If you really want to go ahead,
                please rerun the function with max_len=None.
                """
                )
            )

        # Handle title.
        if title is True:
            title = df.attrs.get("title", None)

        # Indexing.
        if index is None:
            index = list(df.index.names)
        df = df.reset_index().copy()
        if isinstance(index, list):
            index_col = (
                df[index]
                .astype(str)
                .apply(
                    lambda row: ", ".join([o for o in row if o is not None]),
                    axis="columns",
                )
            )
        else:
            assert isinstance(index, str)
            index_col = df[index].astype(str)

        # Check that index is unique.
        if not index_col.is_unique:
            raise ValueError(f"{index} does not produce a unique index")

        # Drop and re-order columns.
        frq_cols = [col for col in df.columns if col.startswith("frq_")]

        # Keep only freq cols.
        heatmap_df = df[frq_cols].copy()

        # Set index.
        heatmap_df.set_index(index_col, inplace=True)

        # Clean column names.
        heatmap_df.columns = heatmap_df.columns.str.lstrip("frq_")

        # Deal with width and height.
        if width is None:
            width = 400 + col_width * len(heatmap_df.columns)
            if colorbar:
                width += 40
        if height is None:
            height = 200 + row_height * len(heatmap_df)
            if title is not None:
                height += 40

        # Plotly heatmap styling.
        fig = px.imshow(
            img=heatmap_df,
            zmin=0,
            zmax=1,
            width=width,
            height=height,
            text_auto=text_auto,
            aspect=aspect,
            color_continuous_scale=color_continuous_scale,
            title=title,
            **kwargs,
        )

        fig.update_xaxes(side="bottom", tickangle=30)
        if x_label is not None:
            fig.update_xaxes(title=x_label)
        if y_label is not None:
            fig.update_yaxes(title=y_label)
        fig.update_layout(
            coloraxis_colorbar=dict(
                title="Frequency",
                tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
                ticktext=["0%", "20%", "40%", "60%", "80%", "100%"],
            )
        )
        if not colorbar:
            fig.update(layout_coloraxis_showscale=False)

        if show:  # pragma: no cover
            fig.show(renderer=renderer)
            return None
        else:
            return fig

    @check_types
    @doc(
        summary="Create a time series plot of variant frequencies using plotly.",
        parameters=dict(
            ds="""
                A dataset of variant frequencies, such as returned by
                `snp_allele_frequencies_advanced()`,
                `aa_allele_frequencies_advanced()` or
                `gene_cnv_frequencies_advanced()`.
            """,
            kwargs="Passed through to `px.line()`.",
        ),
        returns="""
            A plotly figure containing line graphs. The resulting figure will
            have one panel per cohort, grouped into columns by taxon, and
            grouped into rows by area. Markers and lines show frequencies of
            variants.
        """,
    )
    def plot_frequencies_time_series(
        self,
        ds: xr.Dataset,
        height: plotly_params.fig_height = None,
        width: plotly_params.fig_width = None,
        title: plotly_params.title = True,
        legend_sizing: plotly_params.legend_sizing = "constant",
        show: plotly_params.show = True,
        renderer: plotly_params.renderer = None,
        **kwargs,
    ) -> plotly_params.figure:
        # Handle title.
        if title is True:
            title = ds.attrs.get("title", None)

        # Extract cohorts into a dataframe.
        cohort_vars = [v for v in ds if str(v).startswith("cohort_")]
        df_cohorts = ds[cohort_vars].to_dataframe()
        df_cohorts.columns = [c.split("cohort_")[1] for c in df_cohorts.columns]  # type: ignore

        # Extract variant labels.
        variant_labels = ds["variant_label"].values

        # Build a long-form dataframe from the dataset.
        dfs = []
        for cohort_index, cohort in enumerate(df_cohorts.itertuples()):
            ds_cohort = ds.isel(cohorts=cohort_index)
            df = pd.DataFrame(
                {
                    "taxon": cohort.taxon,
                    "area": cohort.area,
                    "date": cohort.period_start,
                    "period": str(
                        cohort.period
                    ),  # use string representation for hover label
                    "sample_size": cohort.size,
                    "variant": variant_labels,
                    "count": ds_cohort["event_count"].values,
                    "nobs": ds_cohort["event_nobs"].values,
                    "frequency": ds_cohort["event_frequency"].values,
                    "frequency_ci_low": ds_cohort["event_frequency_ci_low"].values,
                    "frequency_ci_upp": ds_cohort["event_frequency_ci_upp"].values,
                }
            )
            dfs.append(df)
        df_events = pd.concat(dfs, axis=0).reset_index(drop=True)

        # Remove events with no observations.
        df_events = df_events.query("nobs > 0").copy()

        # Calculate error bars.
        frq = df_events["frequency"]
        frq_ci_low = df_events["frequency_ci_low"]
        frq_ci_upp = df_events["frequency_ci_upp"]
        df_events["frequency_error"] = frq_ci_upp - frq
        df_events["frequency_error_minus"] = frq - frq_ci_low

        # Make a plot.
        fig = px.line(
            df_events,
            facet_col="taxon",
            facet_row="area",
            x="date",
            y="frequency",
            error_y="frequency_error",
            error_y_minus="frequency_error_minus",
            color="variant",
            markers=True,
            hover_name="variant",
            hover_data={
                "frequency": ":.0%",
                "period": True,
                "area": True,
                "taxon": True,
                "sample_size": True,
                "date": False,
                "variant": False,
            },
            height=height,
            width=width,
            title=title,
            labels={
                "date": "Date",
                "frequency": "Frequency",
                "variant": "Variant",
                "taxon": "Taxon",
                "area": "Area",
                "period": "Period",
                "sample_size": "Sample size",
            },
            **kwargs,
        )

        # Tidy plot.
        fig.update_layout(
            yaxis_range=[-0.05, 1.05],
            legend=dict(itemsizing=legend_sizing, tracegroupgap=0),
        )

        if show:  # pragma: no cover
            fig.show(renderer=renderer)
            return None
        else:
            return fig

    @check_types
    @doc(
        summary="""
            Plot markers on a map showing variant frequencies for cohorts grouped
            by area (space), period (time) and taxon.
        """,
        parameters=dict(
            m="The map on which to add the markers.",
            variant="Index or label of variant to plot.",
            taxon="Taxon to show markers for.",
            period="Time period to show markers for.",
            clear="""
                If True, clear all layers (except the base layer) from the map
                before adding new markers.
            """,
        ),
    )
    def plot_frequencies_map_markers(
        self,
        m,
        ds: frq_params.ds_frequencies_advanced,
        variant: Union[int, str],
        taxon: str,
        period: pd.Period,
        clear: bool = True,
    ):
        # Only import here because of some problems importing globally.
        import ipyleaflet  # type: ignore
        import ipywidgets  # type: ignore

        # Slice dataset to variant of interest.
        if isinstance(variant, int):
            ds_variant = ds.isel(variants=variant)
            variant_label = ds["variant_label"].values[variant]
        else:
            assert isinstance(variant, str)
            ds_variant = ds.set_index(variants="variant_label").sel(variants=variant)
            variant_label = variant

        # Convert to a dataframe for convenience.
        df_markers = ds_variant[
            [
                "cohort_taxon",
                "cohort_area",
                "cohort_period",
                "cohort_lat_mean",
                "cohort_lon_mean",
                "cohort_size",
                "event_frequency",
                "event_frequency_ci_low",
                "event_frequency_ci_upp",
            ]
        ].to_dataframe()

        # Select data matching taxon and period parameters.
        df_markers = df_markers.loc[
            (
                (df_markers["cohort_taxon"] == taxon)
                & (df_markers["cohort_period"] == period)
            )
        ]

        # Clear existing layers in the map.
        if clear:
            for layer in m.layers[1:]:
                m.remove_layer(layer)

        # Add markers.
        for x in df_markers.itertuples():
            marker = ipyleaflet.CircleMarker()
            marker.location = (x.cohort_lat_mean, x.cohort_lon_mean)
            marker.radius = 20
            marker.color = "black"
            marker.weight = 1
            marker.fill_color = "red"
            marker.fill_opacity = x.event_frequency
            popup_html = f"""
                <strong>{variant_label}</strong> <br/>
                Taxon: {x.cohort_taxon} <br/>
                Area: {x.cohort_area} <br/>
                Period: {x.cohort_period} <br/>
                Sample size: {x.cohort_size} <br/>
                Frequency: {x.event_frequency:.0%}
                (95% CI: {x.event_frequency_ci_low:.0%} - {x.event_frequency_ci_upp:.0%})
            """
            marker.popup = ipyleaflet.Popup(
                child=ipywidgets.HTML(popup_html),
            )
            m.add(marker)

    @check_types
    @doc(
        summary="""
            Create an interactive map with markers showing variant frequencies or
            cohorts grouped by area (space), period (time) and taxon.
        """,
        parameters=dict(
            title="""
                If True, attempt to use metadata from input dataset as a plot
                title. Otherwise, use supplied value as a title.
            """,
            epilogue="Additional text to display below the map.",
        ),
        returns="""
            An interactive map with widgets for selecting which variant, taxon
            and time period to display.
        """,
    )
    def plot_frequencies_interactive_map(
        self,
        ds: frq_params.ds_frequencies_advanced,
        center: map_params.center = map_params.center_default,
        zoom: map_params.zoom = map_params.zoom_default,
        title: Union[bool, str] = True,
        epilogue: Union[bool, str] = True,
    ):
        import ipyleaflet
        import ipywidgets

        # Handle title.
        if title is True:
            title = ds.attrs.get("title", None)

        # Create a map.
        freq_map = ipyleaflet.Map(center=center, zoom=zoom)

        # Set up interactive controls.
        variants = ds["variant_label"].values
        taxa = np.unique(ds["cohort_taxon"].values)
        periods = np.unique(ds["cohort_period"].values)
        controls = ipywidgets.interactive(
            self.plot_frequencies_map_markers,
            m=ipywidgets.fixed(freq_map),
            ds=ipywidgets.fixed(ds),
            variant=ipywidgets.Dropdown(options=variants, description="Variant: "),
            taxon=ipywidgets.Dropdown(options=taxa, description="Taxon: "),
            period=ipywidgets.Dropdown(options=periods, description="Period: "),
            clear=ipywidgets.fixed(True),
        )

        # Lay out widgets.
        components = []
        if title is not None:
            components.append(ipywidgets.HTML(value=f"<h3>{title}</h3>"))
        components.append(controls)
        components.append(freq_map)
        if epilogue is True:
            epilogue = """
                Variant frequencies are shown as coloured markers. Opacity of color
                denotes frequency. Click on a marker for more information.
            """
        if epilogue:
            components.append(ipywidgets.HTML(value=f"{epilogue}"))

        out = ipywidgets.VBox(components)

        return out

    def snp_genotype_allele_counts(
        self,
        transcript: base_params.transcript,
        snp_query: Optional[base_params.snp_query] = AA_CHANGE_QUERY,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        site_mask: Optional[base_params.site_mask] = None,
    ) -> pd.DataFrame:
        ds_snp = self.snp_calls(
            region=transcript,
            sample_query=sample_query,
            sample_sets=sample_sets,
            site_mask=None,
        )

        # Early check for no SNPs.
        if ds_snp.sizes["variants"] == 0:  # pragma: no cover
            raise ValueError("No SNPs available for the given region and site mask.")

        # Access genotypes.
        gt = ds_snp["call_genotype"].data
        with self._dask_progress(desc="Load SNP genotypes"):
            gt = allel.GenotypeArray(gt.compute())

        # Set up initial dataframe of SNPs.
        df_snps = self._snp_df_melt(ds_snp=ds_snp)

        # Get allele counts.
        gt_counts = gt.to_allele_counts()
        gt_counts_melt = _melt_gt_counts(gt_counts.values)

        df_counts = pd.DataFrame(
            gt_counts_melt, columns=["count_" + s for s in ds_snp["sample_id"].values]
        )
        df_snps = pd.concat([df_snps, df_counts], axis=1)

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

        if site_mask is not None:
            loc_sites = df_snps[f"pass_{site_mask}"]
            df_snps = df_snps.loc[loc_sites]

        return df_snps.query(snp_query)


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


def _make_sample_period_month(row):
    year = row.year
    month = row.month
    if year > 0 and month > 0:
        return pd.Period(freq="M", year=year, month=month)
    else:
        return pd.NaT


def _make_sample_period_quarter(row):
    year = row.year
    month = row.month
    if year > 0 and month > 0:
        return pd.Period(freq="Q", year=year, month=month)
    else:
        return pd.NaT


def _make_sample_period_year(row):
    year = row.year
    if year > 0:
        return pd.Period(freq="Y", year=year)
    else:
        return pd.NaT


def _prep_samples_for_cohort_grouping(*, df_samples, area_by, period_by):
    # Take a copy, as we will modify the dataframe.
    df_samples = df_samples.copy()

    # Fix "intermediate" or "unassigned" taxon values - we only want to build
    # cohorts with clean taxon calls, so we set other values to None.
    loc_intermediate_taxon = (
        df_samples["taxon"].str.startswith("intermediate").fillna(False)
    )
    df_samples.loc[loc_intermediate_taxon, "taxon"] = None
    loc_unassigned_taxon = (
        df_samples["taxon"].str.startswith("unassigned").fillna(False)
    )
    df_samples.loc[loc_unassigned_taxon, "taxon"] = None

    # Add period column.
    if period_by == "year":
        make_period = _make_sample_period_year
    elif period_by == "quarter":
        make_period = _make_sample_period_quarter
    elif period_by == "month":
        make_period = _make_sample_period_month
    else:  # pragma: no cover
        raise ValueError(
            f"Value for period_by parameter must be one of 'year', 'quarter', 'month'; found {period_by!r}."
        )
    sample_period = df_samples.apply(make_period, axis="columns")
    df_samples["period"] = sample_period

    # Add area column for consistent output.
    df_samples["area"] = df_samples[area_by]

    return df_samples


def _build_cohorts_from_sample_grouping(*, group_samples_by_cohort, min_cohort_size):
    # Build cohorts dataframe.
    df_cohorts = group_samples_by_cohort.agg(
        size=("sample_id", len),
        lat_mean=("latitude", "mean"),
        lat_max=("latitude", "max"),
        lat_min=("latitude", "min"),
        lon_mean=("longitude", "mean"),
        lon_max=("longitude", "max"),
        lon_min=("longitude", "min"),
    )
    # Reset index so that the index fields are included as columns.
    df_cohorts = df_cohorts.reset_index()

    # Add cohort helper variables.
    cohort_period_start = df_cohorts["period"].apply(lambda v: v.start_time)
    cohort_period_end = df_cohorts["period"].apply(lambda v: v.end_time)
    df_cohorts["period_start"] = cohort_period_start
    df_cohorts["period_end"] = cohort_period_end
    # Create a label that is similar to the cohort metadata,
    # although this won't be perfect.
    df_cohorts["label"] = df_cohorts.apply(
        lambda v: f"{v.area}_{v.taxon[:4]}_{v.period}", axis="columns"
    )

    # Apply minimum cohort size.
    df_cohorts = df_cohorts.query(f"size >= {min_cohort_size}").reset_index(drop=True)

    return df_cohorts


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


def _add_frequency_ci(*, ds, ci_method):
    from statsmodels.stats.proportion import proportion_confint  # type: ignore

    if ci_method is not None:
        count = ds["event_count"].values
        nobs = ds["event_nobs"].values
        with np.errstate(divide="ignore", invalid="ignore"):
            frq_ci_low, frq_ci_upp = proportion_confint(
                count=count, nobs=nobs, method=ci_method
            )
        ds["event_frequency_ci_low"] = ("variants", "cohorts"), frq_ci_low
        ds["event_frequency_ci_upp"] = ("variants", "cohorts"), frq_ci_upp


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
        # Keep everything as-is, no need for aggregation.
        ds_out = ds[keep_vars + ["variant_alt_allele", "event_count"]]

    else:
        # Take the first value from all variants variables.
        ds_out = ds[keep_vars].isel(variants=[0])

        # Sum event count over variants.
        count = ds["event_count"].values.sum(axis=0, keepdims=True)
        ds_out["event_count"] = ("variants", "cohorts"), count

        # Collapse alt allele.
        alt_allele = "{" + ",".join(ds["variant_alt_allele"].values) + "}"
        ds_out["variant_alt_allele"] = (
            "variants",
            np.array([alt_allele], dtype=object),
        )

    return ds_out
