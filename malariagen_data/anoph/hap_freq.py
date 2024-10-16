from typing import Optional  # , Dict, Union, Callable, List

import pandas as pd
import numpy as np
import xarray as xr
from hashlib import sha1
from numpydoc_decorator import doc  # type: ignore

from ..util import check_types
from .hap_data import AnophelesHapData
from .sample_metadata import locate_cohorts
from . import base_params, frq_params  # , map_params, plotly_params


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

        # Set up cache variables.
        self._cache_annotator = None

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
    def haplotype_frequencies(
        self,
        region: base_params.region,
        cohorts: base_params.cohorts,
        sample_query: Optional[base_params.sample_query] = None,
        min_cohort_size: base_params.min_cohort_size = 10,
        sample_sets: Optional[base_params.sample_sets] = None,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
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
        ds_hap = self.haplotypes(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            chunks=chunks,
            inline_array=inline_array,
        )

        # Early check for no SNPs.
        if ds_hap.sizes["variants"] == 0:  # pragma: no cover
            raise ValueError("No SNPs available for the given region.")

        # Access genotypes.
        gt = ds_hap["call_genotype"].data
        gt = gt.compute()

        # Count haplotypes.
        count_rows: dict[str, int] = dict()
        freq_rows = dict()
        freq_cols = dict()
        cohorts_iterator = coh_dict.items()
        for coh, loc_coh in cohorts_iterator:
            count_rows = {k: 0 for k in count_rows.keys()}
            n_samples = np.count_nonzero(loc_coh)
            assert n_samples >= min_cohort_size
            gt_coh = np.compress(loc_coh, gt, axis=1).copy(order="C")
            for i in range(0, n_samples):
                for j in range(0, 2):
                    gt_cont = np.ascontiguousarray(gt_coh[:, i, j])
                    hap_hash = str(sha1(gt_cont).digest())
                    if hap_hash not in count_rows.keys():
                        count_rows[hap_hash] = 1
                    else:
                        count_rows[hap_hash] += 1
            freq_rows = {k: i / (2 * n_samples) for k, i in count_rows.items()}
            freq_cols["frq_" + coh] = list(freq_rows.values())

        n_haps = np.max([len(i) for i in freq_cols.values()])
        freq_cols = {
            k: v + [0 for i in range(0, n_haps - len(v))] for k, v in freq_cols.items()
        }
        df_freqs = pd.DataFrame(freq_cols, index=freq_rows.keys())

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
    def haplotype_frequencies_advanced(
        self,
        region: base_params.region,
        area_by: frq_params.area_by,
        period_by: frq_params.period_by,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        min_cohort_size: base_params.min_cohort_size = 10,
        ci_method: Optional[frq_params.ci_method] = frq_params.ci_method_default,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
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
        ds_haps = self.haplotypes(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            chunks=chunks,
            inline_array=inline_array,
        )

        # Early check for no SNPs.
        if ds_haps.sizes["variants"] == 0:  # pragma: no cover
            raise ValueError("No SNPs available for the given region.")

        # Access genotypes.
        gt = ds_haps["call_genotype"].data
        gt = gt.compute()

        # Count haplotypes.
        count_rows: dict[str, int] = dict()
        freq_rows = dict()
        freq_cols = dict()
        count_cols = dict()
        nobs_cols = dict()
        cohorts_iterator = enumerate(df_cohorts.itertuples())
        for coh, cohort in cohorts_iterator:
            cohort_key = cohort.taxon, cohort.area, cohort.period
            cohort_key_str = cohort.taxon + "_" + cohort.area + "_" + str(cohort.period)
            count_rows = {k: 0 for k in count_rows.keys()}
            n_samples = cohort.size
            assert n_samples >= min_cohort_size
            sample_indices = group_samples_by_cohort.indices[cohort_key]
            loc_coh = [i in sample_indices for i in range(0, gt.shape[1])]
            gt_coh = np.compress(loc_coh, gt, axis=1)
            for i in range(0, n_samples):
                for j in range(0, 2):
                    gt_cont = np.ascontiguousarray(gt_coh[:, i, j])
                    hap_hash = str(sha1(gt_cont).digest())
                    if hap_hash not in count_rows.keys():
                        count_rows[hap_hash] = 1
                    else:
                        count_rows[hap_hash] += 1
            freq_rows = {k: i / (2 * n_samples) for k, i in count_rows.items()}
            count_cols["count_" + cohort_key_str] = list(count_rows.values())
            freq_cols["frq_" + cohort_key_str] = list(freq_rows.values())
            nobs_cols["nobs_" + cohort_key_str] = [2 * n_samples] * len(freq_rows)

        n_haps = np.max([len(i) for i in freq_cols.values()])
        freq_cols = {
            k: v + [0 for i in range(0, n_haps - len(v))] for k, v in freq_cols.items()
        }
        df_freqs = pd.DataFrame(freq_cols, index=freq_rows.keys())

        # Compute max_af.
        df_max_af = pd.DataFrame({"max_af": df_freqs.max(axis=1)})

        count_cols = {
            k: v + [0 for i in range(0, n_haps - len(v))] for k, v in count_cols.items()
        }
        df_counts = pd.DataFrame(count_cols, index=freq_rows.keys())

        nobs_cols = {
            k: v + [0 for i in range(0, n_haps - len(v))] for k, v in nobs_cols.items()
        }
        df_nobs = pd.DataFrame(nobs_cols, index=freq_rows.keys())

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
        _add_frequency_ci(ds=ds_out, ci_method=ci_method)

        return ds_out


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
