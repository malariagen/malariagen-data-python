from typing import Optional, List
from bisect import bisect_left, bisect_right

import numpy as np
import pandas as pd
import xarray as xr
import dask
import numba
import warnings
from numpydoc_decorator import doc  # type: ignore

from . import base_params, cnv_params, frq_params
from ..util import (
    check_types,
    pandas_apply,
    Region,
    parse_multi_region,
    region_str,
    simple_xarray_concat,
    prep_samples_for_cohort_grouping,
    build_cohorts_from_sample_grouping,
    add_frequency_ci,
)
from .cnv_data import AnophelesCnvData
from .sample_metadata import locate_cohorts


class AnophelesCnvFrequencyAnalysis(
    AnophelesCnvData,
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
        Compute modal copy number by gene, from HMM data.
        """,
        returns="""
        A dataset of modal copy number per gene and associated data.
        """,
    )
    def gene_cnv(
        self,
        region: base_params.regions,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        max_coverage_variance: cnv_params.max_coverage_variance = cnv_params.max_coverage_variance_default,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> xr.Dataset:
        regions: List[Region] = parse_multi_region(self, region)
        del region

        ds = simple_xarray_concat(
            [
                self._gene_cnv(
                    region=r,
                    sample_sets=sample_sets,
                    sample_query=sample_query,
                    sample_query_options=sample_query_options,
                    max_coverage_variance=max_coverage_variance,
                    chunks=chunks,
                    inline_array=inline_array,
                )
                for r in regions
            ],
            dim="genes",
        )

        return ds

    def _gene_cnv(
        self,
        *,
        region,
        sample_sets,
        sample_query,
        sample_query_options,
        max_coverage_variance,
        chunks,
        inline_array,
    ):
        # Sanity check.
        assert isinstance(region, Region)

        # Access genes within the region of interest.
        df_genome_features = self.genome_features(region=region)
        sample_query_options = sample_query_options or {}
        df_genes = df_genome_features.query(
            f"type == '{self._gff_gene_type}'", **sample_query_options
        )

        # Refine the region for CNV data to ensure coverage of all requested genes.
        cnv_region = Region(
            region.contig, df_genes["start"].min(), df_genes["end"].max()
        )

        # Access HMM data.
        ds_hmm = self.cnv_hmm(
            region=cnv_region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            max_coverage_variance=max_coverage_variance,
            chunks=chunks,
            inline_array=inline_array,
        )
        pos = ds_hmm["variant_position"].data
        end = ds_hmm["variant_end"].data
        cn = ds_hmm["call_CN"].data.astype("int8", casting="same_kind")
        with self._dask_progress(desc="Load CNV HMM data"):
            pos, end, cn = dask.compute(pos, end, cn)

        # Set up intermediates.
        windows = []
        modes = []
        counts = []

        # Iterate over genes.
        genes_iterator = self._progress(
            df_genes.itertuples(),
            desc="Compute modal gene copy number",
            total=len(df_genes),
        )
        for gene in genes_iterator:
            # Locate windows overlapping the gene.
            loc_gene_start = bisect_left(end, gene.start)
            loc_gene_stop = bisect_right(pos, gene.end)
            w = loc_gene_stop - loc_gene_start
            windows.append(w)

            # Slice out copy number data for the given gene.
            cn_gene = cn[loc_gene_start:loc_gene_stop]

            # Compute the modes.
            m, c = _cn_mode(cn_gene, vmax=12)
            modes.append(m)
            counts.append(c)

        # Combine results.
        windows = np.array(windows)
        modes = np.vstack(modes)
        counts = np.vstack(counts)

        # Build dataset.
        ds_out = xr.Dataset(
            coords={
                "gene_id": (["genes"], df_genes["ID"].values),
                "sample_id": (["samples"], ds_hmm["sample_id"].values),
            },
            data_vars={
                "gene_contig": (["genes"], df_genes["contig"].values),
                "gene_start": (["genes"], df_genes["start"].values),
                "gene_end": (["genes"], df_genes["end"].values),
                "gene_windows": (["genes"], windows),
                "gene_name": (
                    ["genes"],
                    df_genes[self._gff_gene_name_attribute].values,
                ),
                "gene_strand": (["genes"], df_genes["strand"].values),
                "gene_description": (["genes"], df_genes["description"].values),
                "CN_mode": (["genes", "samples"], modes),
                "CN_mode_count": (["genes", "samples"], counts),
                "sample_coverage_variance": (
                    ["samples"],
                    ds_hmm["sample_coverage_variance"].values,
                ),
                "sample_is_high_variance": (
                    ["samples"],
                    ds_hmm["sample_is_high_variance"].values,
                ),
            },
        )

        return ds_out

    @check_types
    @doc(
        summary="""
        Compute modal copy number by gene, then compute the frequency of
        amplifications and deletions in one or more cohorts, from HMM data.
        """,
        returns="""
            A dataframe of CNV amplification (amp) and deletion (del)
            frequencies in the specified cohorts, one row per gene and CNV type
            (amp/del).
        """,
    )
    def gene_cnv_frequencies(
        self,
        region: base_params.regions,
        cohorts: base_params.cohorts,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        min_cohort_size: base_params.min_cohort_size = 10,
        max_coverage_variance: cnv_params.max_coverage_variance = cnv_params.max_coverage_variance_default,
        sample_sets: Optional[base_params.sample_sets] = None,
        drop_invariant: frq_params.drop_invariant = True,
        include_counts: frq_params.include_counts = False,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> pd.DataFrame:
        debug = self._log.debug

        debug("check and normalise parameters")
        regions: List[Region] = parse_multi_region(self, region)
        del region

        debug("access and concatenate data from regions")
        df = pd.concat(
            [
                self._gene_cnv_frequencies(
                    region=r,
                    cohorts=cohorts,
                    sample_query=sample_query,
                    sample_query_options=sample_query_options,
                    min_cohort_size=min_cohort_size,
                    sample_sets=sample_sets,
                    drop_invariant=drop_invariant,
                    max_coverage_variance=max_coverage_variance,
                    include_counts=include_counts,
                    chunks=chunks,
                    inline_array=inline_array,
                )
                for r in regions
            ],
            axis=0,
        )

        debug("add metadata")
        title = f"Gene CNV frequencies ({region_str(regions)})"
        df.attrs["title"] = title

        return df

    def _gene_cnv_frequencies(
        self,
        *,
        region,
        cohorts,
        sample_query,
        sample_query_options,
        min_cohort_size,
        sample_sets,
        drop_invariant,
        max_coverage_variance,
        include_counts,
        chunks,
        inline_array,
    ):
        debug = self._log.debug

        debug("sanity check - this function is one region at a time")
        assert isinstance(region, Region)

        debug("get gene copy number data")
        ds_cnv = self.gene_cnv(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            max_coverage_variance=max_coverage_variance,
            chunks=chunks,
            inline_array=inline_array,
        )

        debug("load sample metadata")
        df_samples = self.sample_metadata(sample_sets=sample_sets)

        debug("align sample metadata with samples in CNV data")
        sample_id = ds_cnv["sample_id"].values
        df_samples = df_samples.set_index("sample_id").loc[sample_id].reset_index()

        debug("figure out expected copy number")
        if region.contig == "X":
            is_male = (df_samples["sex_call"] == "M").values
            expected_cn = np.where(is_male, 1, 2)[np.newaxis, :]
        else:
            expected_cn = 2

        debug(
            "setup output dataframe - two rows for each gene, one for amplification and one for deletion"
        )
        n_genes = ds_cnv.sizes["genes"]
        df_genes = ds_cnv[
            [
                "gene_id",
                "gene_name",
                "gene_strand",
                "gene_description",
                "gene_contig",
                "gene_start",
                "gene_end",
            ]
        ].to_dataframe()
        df = pd.concat([df_genes, df_genes], axis=0).reset_index(drop=True)
        df.rename(
            columns={
                "gene_contig": "contig",
                "gene_start": "start",
                "gene_end": "end",
            },
            inplace=True,
        )

        debug("add CNV type column")
        df_cnv_type = pd.DataFrame(
            {
                "cnv_type": np.array(
                    (["amp"] * n_genes) + (["del"] * n_genes), dtype=object
                )
            }
        )
        df = pd.concat([df, df_cnv_type], axis=1)

        debug("set up intermediates")
        cn = ds_cnv["CN_mode"].values
        is_amp = cn > expected_cn
        is_del = (cn >= 0) & (cn < expected_cn)
        is_called = cn >= 0

        debug("set up cohort dict")
        coh_dict = locate_cohorts(
            cohorts=cohorts, data=df_samples, min_cohort_size=min_cohort_size
        )

        if len(coh_dict) == 0:
            raise ValueError(
                "No cohorts available for the given sample selection parameters and minimum cohort size."
            )

        debug("compute cohort frequencies")
        freq_cols = dict()
        count_cols = dict()
        nobs_cols = dict()
        for coh, loc_coh in coh_dict.items():
            n_samples = np.count_nonzero(loc_coh)
            debug(f"{coh}, {n_samples} samples")

            if n_samples >= min_cohort_size:
                # subset data to cohort
                is_amp_coh = np.compress(loc_coh, is_amp, axis=1)
                is_del_coh = np.compress(loc_coh, is_del, axis=1)
                is_called_coh = np.compress(loc_coh, is_called, axis=1)

                # count amplifications and deletions
                amp_count_coh = np.sum(is_amp_coh, axis=1)
                del_count_coh = np.sum(is_del_coh, axis=1)
                called_count_coh = np.sum(is_called_coh, axis=1)

                # compute frequencies, taking accessibility into account
                with np.errstate(divide="ignore", invalid="ignore"):
                    amp_freq_coh = np.where(
                        called_count_coh > 0, amp_count_coh / called_count_coh, np.nan
                    )
                    del_freq_coh = np.where(
                        called_count_coh > 0, del_count_coh / called_count_coh, np.nan
                    )
                nobs_cols[f"nobs_{coh}"] = np.concatenate(
                    [called_count_coh, called_count_coh]
                )
                count_cols[f"count_{coh}"] = np.concatenate(
                    [amp_count_coh, del_count_coh]
                )
                freq_cols[f"frq_{coh}"] = np.concatenate([amp_freq_coh, del_freq_coh])

        debug("build a dataframe with the frequency columns")
        df_nobs = pd.DataFrame(nobs_cols)
        df_counts = pd.DataFrame(count_cols)
        df_freqs = pd.DataFrame(freq_cols)

        debug("compute max_af and additional columns")
        df_extras = pd.DataFrame(
            {
                "max_af": df_freqs.max(axis=1),
                "windows": np.concatenate(
                    [ds_cnv["gene_windows"].values, ds_cnv["gene_windows"].values]
                ),
            }
        )

        # Build the final dataframe.
        debug("build the final dataframe")
        df.reset_index(drop=True, inplace=True)
        if include_counts:
            df = pd.concat([df, df_freqs, df_extras, df_counts, df_nobs], axis=1)
        else:
            df = pd.concat([df, df_freqs, df_extras], axis=1)
        df.sort_values(["contig", "start", "cnv_type"], inplace=True)
        df.reset_index(drop=True, inplace=True)

        debug("add label")
        df["label"] = pandas_apply(
            _make_gene_cnv_label, df, columns=["gene_id", "gene_name", "cnv_type"]
        )

        debug("deal with invariants")
        if drop_invariant:
            df = df.query("max_af > 0")

        debug("set index for convenience")
        df.set_index(["gene_id", "gene_name", "cnv_type"], inplace=True)

        return df

    @check_types
    @doc(
        summary="""
        Group samples by taxon, area (space) and period (time), then compute
        gene CNV counts and frequencies.
        """,
        returns="""
            The resulting dataset contains data has dimensions "cohorts" and
            "variants". Variables prefixed with "cohort" are 1-dimensional
            arrays with data about the cohorts, such as the area, period, taxon
            and cohort size. Variables prefixed with "variant" are 1-dimensional
            arrays with data about the variants, such as the contig, position,
            reference and alternate alleles. Variables prefixed with "event" are
            2-dimensional arrays with the allele counts and frequency
            calculations.
        """,
    )
    def gene_cnv_frequencies_advanced(
        self,
        region: base_params.regions,
        area_by: frq_params.area_by,
        period_by: frq_params.period_by,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        min_cohort_size: base_params.min_cohort_size = 10,
        drop_invariant: frq_params.drop_invariant = True,
        variant_query: Optional[frq_params.variant_query] = None,
        max_coverage_variance: cnv_params.max_coverage_variance = cnv_params.max_coverage_variance_default,
        nobs_mode: frq_params.nobs_mode = frq_params.nobs_mode_default,
        ci_method: Optional[frq_params.ci_method] = frq_params.ci_method_default,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ) -> xr.Dataset:
        regions: List[Region] = parse_multi_region(self, region)
        del region

        ds = simple_xarray_concat(
            [
                self._gene_cnv_frequencies_advanced(
                    region=r,
                    area_by=area_by,
                    period_by=period_by,
                    sample_sets=sample_sets,
                    sample_query=sample_query,
                    sample_query_options=sample_query_options,
                    min_cohort_size=min_cohort_size,
                    variant_query=variant_query,
                    drop_invariant=drop_invariant,
                    max_coverage_variance=max_coverage_variance,
                    nobs_mode=nobs_mode,
                    ci_method=ci_method,
                    chunks=chunks,
                    inline_array=inline_array,
                )
                for r in regions
            ],
            dim="variants",
        )

        title = f"Gene CNV frequencies ({region_str(regions)})"
        ds.attrs["title"] = title

        return ds

    def _gene_cnv_frequencies_advanced(
        self,
        *,
        region,
        area_by,
        period_by,
        sample_sets,
        sample_query,
        sample_query_options,
        min_cohort_size,
        variant_query,
        drop_invariant,
        max_coverage_variance,
        nobs_mode,
        ci_method,
        chunks,
        inline_array,
    ):
        debug = self._log.debug

        debug("sanity check - here we deal with one region only")
        assert isinstance(region, Region)

        debug("access gene CNV calls")
        ds_cnv = self.gene_cnv(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            max_coverage_variance=max_coverage_variance,
            chunks=chunks,
            inline_array=inline_array,
        )

        debug("load sample metadata")
        df_samples = self.sample_metadata(sample_sets=sample_sets)

        debug("align sample metadata")
        sample_id = ds_cnv["sample_id"].values
        df_samples = df_samples.set_index("sample_id").loc[sample_id].reset_index()

        debug("prepare sample metadata for cohort grouping")
        df_samples = prep_samples_for_cohort_grouping(
            df_samples=df_samples,
            area_by=area_by,
            period_by=period_by,
        )

        debug("group samples to make cohorts")
        group_samples_by_cohort = df_samples.groupby(["taxon", "area", "period"])

        debug("build cohorts dataframe")
        df_cohorts = build_cohorts_from_sample_grouping(
            group_samples_by_cohort=group_samples_by_cohort,
            min_cohort_size=min_cohort_size,
        )

        debug("figure out expected copy number")
        if region.contig == "X":
            is_male = (df_samples["sex_call"] == "M").values
            expected_cn = np.where(is_male, 1, 2)[np.newaxis, :]
        else:
            expected_cn = 2

        debug("set up intermediates")
        cn = ds_cnv["CN_mode"].values
        is_amp = cn > expected_cn
        is_del = (cn >= 0) & (cn < expected_cn)
        is_called = cn >= 0

        debug("set up main event variables")
        n_genes = ds_cnv.sizes["genes"]
        n_variants, n_cohorts = n_genes * 2, len(df_cohorts)
        count = np.zeros((n_variants, n_cohorts), dtype=int)
        nobs = np.zeros((n_variants, n_cohorts), dtype=int)

        debug("build event count and nobs for each cohort")
        for cohort_index, cohort in enumerate(df_cohorts.itertuples()):
            # construct grouping key
            cohort_key = cohort.taxon, cohort.area, cohort.period

            # obtain sample indices for cohort
            sample_indices = group_samples_by_cohort.indices[cohort_key]

            # select genotype data for cohort
            cohort_is_amp = np.take(is_amp, sample_indices, axis=1)
            cohort_is_del = np.take(is_del, sample_indices, axis=1)
            cohort_is_called = np.take(is_called, sample_indices, axis=1)

            # compute cohort allele counts
            np.sum(cohort_is_amp, axis=1, out=count[::2, cohort_index])
            np.sum(cohort_is_del, axis=1, out=count[1::2, cohort_index])

            # compute cohort allele numbers
            cohort_n_called = np.sum(cohort_is_called, axis=1)
            if nobs_mode == "called":
                nobs[:, cohort_index] = np.repeat(cohort_n_called, 2)
            else:
                assert nobs_mode == "fixed"
                nobs[:, cohort_index] = cohort.size * 2

        debug("compute frequency")
        with np.errstate(divide="ignore", invalid="ignore"):
            # ignore division warnings
            frequency = np.where(nobs > 0, count / nobs, np.nan)

        debug("make dataframe of variants")
        with warnings.catch_warnings():
            # ignore "All-NaN slice encountered" warnings
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_af = np.nanmax(frequency, axis=1)
        df_variants = pd.DataFrame(
            {
                "contig": region.contig,
                "start": np.repeat(ds_cnv["gene_start"].values, 2),
                "end": np.repeat(ds_cnv["gene_end"].values, 2),
                "windows": np.repeat(ds_cnv["gene_windows"].values, 2),
                # alternate amplification and deletion
                "cnv_type": np.tile(np.array(["amp", "del"]), n_genes),
                "max_af": max_af,
                "gene_id": np.repeat(ds_cnv["gene_id"].values, 2),
                "gene_name": np.repeat(ds_cnv["gene_name"].values, 2),
                "gene_strand": np.repeat(ds_cnv["gene_strand"].values, 2),
            }
        )

        debug("add variant label")
        df_variants["label"] = pandas_apply(
            _make_gene_cnv_label,
            df_variants,
            columns=["gene_id", "gene_name", "cnv_type"],
        )

        debug("build the output dataset")
        ds_out = xr.Dataset()

        debug("cohort variables")
        for coh_col in df_cohorts.columns:
            ds_out[f"cohort_{coh_col}"] = "cohorts", df_cohorts[coh_col]

        debug("variant variables")
        for snp_col in df_variants.columns:
            ds_out[f"variant_{snp_col}"] = "variants", df_variants[snp_col]

        debug("event variables")
        ds_out["event_count"] = ("variants", "cohorts"), count
        ds_out["event_nobs"] = ("variants", "cohorts"), nobs
        ds_out["event_frequency"] = ("variants", "cohorts"), frequency

        debug("deal with invariants")
        if drop_invariant:
            loc_variant = df_variants["max_af"].values > 0
            ds_out = ds_out.isel(variants=loc_variant)
            df_variants = df_variants.loc[loc_variant].reset_index(drop=True)

        debug("apply variant query")
        if variant_query is not None:
            loc_variants = df_variants.eval(variant_query).values
            ds_out = ds_out.isel(variants=loc_variants)

        debug("add confidence intervals")
        add_frequency_ci(ds=ds_out, ci_method=ci_method)

        debug("tidy up display by sorting variables")
        ds_out = ds_out[sorted(ds_out)]

        return ds_out


def _make_gene_cnv_label(gene_id, gene_name, cnv_type):
    label = gene_id
    if isinstance(gene_name, str):
        label += f" ({gene_name})"
    label += f" {cnv_type}"
    return label


@numba.njit("Tuple((int8, int64))(int8[:], int8)")
def _cn_mode_1d(a, vmax):
    # setup intermediates
    m = a.shape[0]
    counts = np.zeros(vmax + 1, dtype=numba.int64)

    # initialise return values
    mode = numba.int8(-1)
    mode_count = numba.int64(0)

    # iterate over array values, keeping track of counts
    for i in range(m):
        v = a[i]
        if 0 <= v <= vmax:
            c = counts[v]
            c += 1
            counts[v] = c
            if c > mode_count:
                mode = v
                mode_count = c
            elif c == mode_count and v < mode:
                # consistency with scipy.stats, break ties by taking lower value
                mode = v

    return mode, mode_count


@numba.njit("Tuple((int8[:], int64[:]))(int8[:, :], int8)")
def _cn_mode(a, vmax):
    # setup intermediates
    n = a.shape[1]

    # setup outputs
    modes = np.zeros(n, dtype=numba.int8)
    counts = np.zeros(n, dtype=numba.int64)

    # iterate over columns, computing modes
    for j in range(a.shape[1]):
        mode, count = _cn_mode_1d(a[:, j], vmax)
        modes[j] = mode
        counts[j] = count

    return modes, counts
