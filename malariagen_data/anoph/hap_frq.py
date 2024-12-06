from typing import Optional

import pandas as pd
import numpy as np
import xarray as xr
import allel
from numpydoc_decorator import doc  # type: ignore

from ..util import (
    check_types,
    haplotype_frequencies,
    prep_samples_for_cohort_grouping,
    build_cohorts_from_sample_grouping,
    add_frequency_ci,
)
from .hap_data import AnophelesHapData
from .sample_metadata import locate_cohorts
from . import base_params, frq_params


class AnophelesHapFrequencyAnalysis(
    AnophelesHapData,
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
        summary="""
            Compute haplotype frequencies for a region.
        """,
        returns="""
            A dataframe of haplotype frequencies, one row per haplotype.
        """,
        notes="""
            Cohorts with fewer samples than `min_cohort_size` will be excluded from
            output data frame.
        """,
    )
    def haplotypes_frequencies(
        self,
        region: base_params.region,
        cohorts: base_params.cohorts,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        min_cohort_size: base_params.min_cohort_size = 10,
        sample_sets: Optional[base_params.sample_sets] = None,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> pd.DataFrame:
        # Access sample metadata.
        df_samples = self.sample_metadata(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
        )

        # Build cohort dictionary, maps cohort labels to boolean indexers.
        coh_dict = locate_cohorts(
            cohorts=cohorts, data=df_samples, min_cohort_size=min_cohort_size
        )

        # Access haplotypes.
        ds_haps = self.haplotypes(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            chunks=chunks,
            inline_array=inline_array,
        )

        # Early check for no SNPs.
        if ds_haps.sizes["variants"] == 0:  # pragma: no cover
            raise ValueError("No SNPs available for the given region.")

        # Access genotypes.
        gt = allel.GenotypeDaskArray(ds_haps["call_genotype"].data)
        with self._dask_progress(desc="Compute haplotypes"):
            gt = gt.compute()

        # List all haplotypes
        gt_hap = gt.to_haplotypes()
        f_all, _, _ = haplotype_frequencies(gt_hap)

        # Count haplotypes.
        freq_cols = dict()
        cohorts_iterator = self._progress(
            coh_dict.items(), desc="Compute allele frequencies"
        )
        for coh, loc_coh in cohorts_iterator:
            # We reset all frequencies to 0 for each cohort
            hap_dict = {k: 0 for k in f_all.keys()}

            n_samples = np.count_nonzero(loc_coh)
            assert n_samples >= min_cohort_size
            gt_coh = gt.compress(loc_coh, axis=1)
            gt_hap = gt_coh.to_haplotypes()
            f, _, _ = haplotype_frequencies(gt_hap)
            # The frequencies of the observed haplotypes are then updated
            hap_dict.update(f)
            freq_cols["frq_" + coh] = list(hap_dict.values())

        df_freqs = pd.DataFrame(freq_cols, index=list(hap_dict.keys()))

        # Compute max_af.
        df_max_af = pd.DataFrame({"max_af": df_freqs.max(axis=1)})

        # Build the final dataframe.
        df_haps = pd.concat([df_freqs, df_max_af], axis=1)

        df_haps_sorted = df_haps.sort_values(by=["max_af"], ascending=False)
        df_haps_sorted["label"] = ["H" + str(i) for i in range(len(df_haps_sorted))]

        # Reset index after filtering.
        df_haps_sorted.set_index(keys="label", drop=True)

        return df_haps_sorted

    @check_types
    @doc(
        summary="""
            Group samples by taxon, area (space) and period (time), then compute
            haplotype frequencies.
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
    def haplotypes_frequencies_advanced(
        self,
        region: base_params.region,
        area_by: frq_params.area_by,
        period_by: frq_params.period_by,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        min_cohort_size: base_params.min_cohort_size = 10,
        ci_method: Optional[frq_params.ci_method] = frq_params.ci_method_default,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> xr.Dataset:
        # Load sample metadata.
        df_samples = self.sample_metadata(
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
        )

        # Prepare sample metadata for cohort grouping.
        df_samples = prep_samples_for_cohort_grouping(
            df_samples=df_samples,
            area_by=area_by,
            period_by=period_by,
        )

        # Group samples to make cohorts.
        group_samples_by_cohort = df_samples.groupby(["taxon", "area", "period"])

        # Build cohorts dataframe.
        df_cohorts = build_cohorts_from_sample_grouping(
            group_samples_by_cohort=group_samples_by_cohort,
            min_cohort_size=min_cohort_size,
        )

        # Access haplotypes.
        ds_haps = self.haplotypes(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            chunks=chunks,
            inline_array=inline_array,
        )

        # Early check for no SNPs.
        if ds_haps.sizes["variants"] == 0:  # pragma: no cover
            raise ValueError("No SNPs available for the given region.")

        # Access genotypes.
        gt = allel.GenotypeDaskArray(ds_haps["call_genotype"].data)
        with self._dask_progress(desc="Compute haplotypes"):
            gt = gt.compute()

        # List all haplotypes
        gt_hap = gt.to_haplotypes()
        f_all, _, _ = haplotype_frequencies(gt_hap)

        # Count haplotypes.
        hap_freq: dict[np.int64, float] = dict()
        hap_count: dict[np.int64, int] = dict()
        hap_nob: dict[np.int64, int] = dict()
        freq_cols = dict()
        count_cols = dict()
        nobs_cols = dict()
        cohorts_iterator = self._progress(
            df_cohorts.itertuples(), desc="Compute allele frequencies"
        )
        for cohort in cohorts_iterator:
            cohort_key = cohort.taxon, cohort.area, cohort.period
            cohort_key_str = cohort.taxon + "_" + cohort.area + "_" + str(cohort.period)
            # We reset all frequencies, counts to 0 for each cohort, nobs is set to the number of haplotypes
            n_samples = cohort.size
            hap_freq = {k: 0 for k in f_all.keys()}
            hap_count = {k: 0 for k in f_all.keys()}
            hap_nob = {k: 2 * n_samples for k in f_all.keys()}
            assert n_samples >= min_cohort_size
            sample_indices = group_samples_by_cohort.indices[cohort_key]
            gt_coh = gt.take(sample_indices, axis=1)
            gt_hap = gt_coh.to_haplotypes()
            f, c, o = haplotype_frequencies(gt_hap)
            # The frequencies and counts of the observed haplotypes are then updated, so are the nobs but the values should actually stay the same
            hap_freq.update(f)
            hap_count.update(c)
            hap_nob.update(o)
            count_cols["count_" + cohort_key_str] = list(hap_count.values())
            freq_cols["frq_" + cohort_key_str] = list(hap_freq.values())
            nobs_cols["nobs_" + cohort_key_str] = list(hap_nob.values())

        df_freqs = pd.DataFrame(freq_cols, index=list(hap_freq.keys()))

        # Compute max_af.
        df_max_af = pd.DataFrame({"max_af": df_freqs.max(axis=1)})

        df_counts = pd.DataFrame(count_cols, index=list(hap_count.keys()))

        df_nobs = pd.DataFrame(nobs_cols, index=list(hap_nob.keys()))

        # Build the final dataframe.
        df_haps = pd.concat([df_freqs, df_counts, df_nobs, df_max_af], axis=1)

        df_haps_sorted = df_haps.sort_values(by=["max_af"], ascending=False)
        df_haps_sorted["label"] = ["H" + str(i) for i in range(len(df_haps_sorted))]

        # Reset index after filtering.
        df_haps_sorted.set_index(keys="label", drop=True)

        # Build the output dataset.
        ds_out = xr.Dataset()

        # Cohort variables.
        for coh_col in df_cohorts.columns:
            ds_out[f"cohort_{coh_col}"] = "cohorts", df_cohorts[coh_col]

        # Label the haplotypes
        ds_out["variant_label"] = "variants", df_haps_sorted["label"]
        # Event variables.
        ds_out["event_frequency"] = (
            ("variants", "cohorts"),
            df_haps_sorted.to_numpy()[:, : len(df_cohorts)],
        )
        ds_out["event_count"] = (
            ("variants", "cohorts"),
            df_haps_sorted.to_numpy()[:, len(df_cohorts) : 2 * len(df_cohorts)],
        )
        ds_out["event_nobs"] = (
            ("variants", "cohorts"),
            df_haps_sorted.to_numpy()[:, 2 * len(df_cohorts) : -2],
        )

        # Add confidence intervals.
        add_frequency_ci(ds=ds_out, ci_method=ci_method)

        return ds_out
