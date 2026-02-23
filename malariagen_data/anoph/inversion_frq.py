from typing import Optional

import numpy as np
import pandas as pd
from numpydoc_decorator import doc  # type: ignore
import xarray as xr

from ..util import _check_types
from .karyotype import AnophelesKaryotypeAnalysis
from .frq_base import (
    AnophelesFrequencyAnalysis,
    _prep_samples_for_cohort_grouping,
    _build_cohorts_from_sample_grouping,
    _add_frequency_ci,
)
from .sample_metadata import _locate_cohorts
from .karyotype_params import inversions_param
from . import base_params, frq_params


AA_CHANGE_QUERY = (
    "effect in ['NON_SYNONYMOUS_CODING', 'START_LOST', 'STOP_LOST', 'STOP_GAINED']"
)


class AnophelesInversionFrequencyAnalysis(
    AnophelesKaryotypeAnalysis, AnophelesFrequencyAnalysis
):
    def __init__(
        self,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

    @_check_types
    @doc(
        summary="""
            Compute inversion frequencies for a sequence of inversions.
        """,
        returns="""
            A dataframe of inversion frequencies, one row per karyotype.
        """,
        notes="""
            Cohorts with fewer samples than `min_cohort_size` will be excluded from
            output data frame.
        """,
    )
    def inversion_frequencies(
        self,
        inversions: inversions_param,
        cohorts: base_params.cohorts,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        min_cohort_size: base_params.min_cohort_size = 10,
        sample_sets: Optional[base_params.sample_sets] = None,
        include_counts: frq_params.include_counts = False,
    ) -> pd.DataFrame:
        if inversions == []:
            raise ValueError("At least one inversion needs to be provided.")
        elif isinstance(inversions, list):
            df_kar_frqs_list = []
            for inversion in inversions:
                # Access sample metadata.
                df_samples = self.sample_metadata(
                    sample_sets=sample_sets,
                    sample_query=sample_query,
                    sample_query_options=sample_query_options,
                )

                # Build cohort dictionary, maps cohort labels to boolean indexers.
                coh_dict = _locate_cohorts(
                    cohorts=cohorts, data=df_samples, min_cohort_size=min_cohort_size
                )

                # Access karyotypes
                kar_df = self.karyotype(
                    inversion=inversion,
                    sample_sets=sample_sets,
                    sample_query=sample_query,
                    sample_query_options=sample_query_options,
                )

                base_df = pd.DataFrame(
                    {
                        "inversion": [inversion] * 3,
                        "allele": ["hom. ref.", "het.", "hom. alt."],
                        "label": [
                            f"{inversion} hom. ref.",
                            f"{inversion} het.",
                            f"{inversion} hom. alt.",
                        ],
                    }
                )

                # Count alleles.
                count_cols = dict()
                nobs_cols = dict()
                freq_cols = dict()
                cohorts_iterator = self._progress(
                    coh_dict.items(), desc="Compute karyotype frequencies"
                )
                for coh, loc_coh in cohorts_iterator:
                    n_samples = np.count_nonzero(loc_coh)
                    assert n_samples >= min_cohort_size
                    kar_loc = kar_df.loc[loc_coh]

                    count_cols[f"count_{coh}"] = [
                        len(kar_loc.query(f"karyotype_{inversion} == {i}"))
                        for i in range(0, 3)
                    ]
                    freq_cols[f"frq_{coh}"] = [
                        c / n_samples for c in count_cols[f"count_{coh}"]
                    ]
                    nobs = 2 * n_samples
                    nobs_cols[f"nobs_{coh}"] = [nobs] * 3

                # Build a dataframe with the frequency columns.
                df_freqs = pd.DataFrame(freq_cols)
                df_counts = pd.DataFrame(count_cols)
                df_nobs = pd.DataFrame(nobs_cols)

                # Build the final dataframe.
                if include_counts:
                    df_kar_frqs_inv = pd.concat(
                        [base_df, df_freqs, df_counts, df_nobs], axis=1
                    )
                else:
                    df_kar_frqs_inv = pd.concat([base_df, df_freqs], axis=1)

                df_kar_frqs_list.append(df_kar_frqs_inv)

            df_kar_frqs = pd.concat(df_kar_frqs_list, axis=0)

            return df_kar_frqs
        else:
            # Access sample metadata.
            df_samples = self.sample_metadata(
                sample_sets=sample_sets,
                sample_query=sample_query,
                sample_query_options=sample_query_options,
            )

            # Build cohort dictionary, maps cohort labels to boolean indexers.
            coh_dict = _locate_cohorts(
                cohorts=cohorts, data=df_samples, min_cohort_size=min_cohort_size
            )

            # Access karyotypes
            kar_df = self.karyotype(
                inversion=inversions,
                sample_sets=sample_sets,
                sample_query=sample_query,
                sample_query_options=sample_query_options,
            )

            base_df = pd.DataFrame(
                {
                    "inversion": [inversions] * 3,
                    "allele": ["hom. ref.", "het.", "hom. alt."],
                    "label": [
                        f"{inversions} hom. ref.",
                        f"{inversions} het.",
                        f"{inversions} hom. alt.",
                    ],
                }
            )

            # Count alleles.
            count_cols = dict()
            nobs_cols = dict()
            freq_cols = dict()
            cohorts_iterator = self._progress(
                coh_dict.items(), desc="Compute karyotype frequencies"
            )
            for coh, loc_coh in cohorts_iterator:
                n_samples = np.count_nonzero(loc_coh)
                assert n_samples >= min_cohort_size
                kar_loc = kar_df.loc[loc_coh]

                count_cols[f"count_{coh}"] = [
                    len(kar_loc.query(f"karyotype_{inversions} == {i}"))
                    for i in range(0, 3)
                ]
                freq_cols[f"frq_{coh}"] = [
                    c / n_samples for c in count_cols[f"count_{coh}"]
                ]
                nobs = 2 * n_samples
                nobs_cols[f"nobs_{coh}"] = [nobs] * 3

            # Build a dataframe with the frequency columns.
            df_freqs = pd.DataFrame(freq_cols)
            df_counts = pd.DataFrame(count_cols)
            df_nobs = pd.DataFrame(nobs_cols)

            # Build the final dataframe.
            if include_counts:
                df_kar_frqs_inv = pd.concat(
                    [base_df, df_freqs, df_counts, df_nobs], axis=1
                )
            else:
                df_kar_frqs_inv = pd.concat([base_df, df_freqs], axis=1)

            return df_kar_frqs_inv

    @_check_types
    @doc(
        summary="""
            Group samples by taxon, area (space) and period (time), then compute
            inversion frequencies.
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
    )  # I need to add the option for inversionS
    def inversion_frequencies_advanced(
        self,
        inversions: inversions_param,
        area_by: frq_params.area_by,
        period_by: frq_params.period_by,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        min_cohort_size: base_params.min_cohort_size = 10,
        ci_method: Optional[frq_params.ci_method] = frq_params.ci_method_default,
    ) -> xr.Dataset:
        if inversions == []:
            raise ValueError("At least one inversion needs to be provided.")
        elif isinstance(inversions, list):
            ds_list = []

            # Load sample metadata.
            df_samples = self.sample_metadata(
                sample_sets=sample_sets,
                sample_query=sample_query,
                sample_query_options=sample_query_options,
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
            for inversion in inversions:
                # Access karyotypes.
                kar_df = self.karyotype(
                    inversion=inversion,
                    sample_sets=sample_sets,
                    sample_query=sample_query,
                    sample_query_options=sample_query_options,
                )

                # Count alleles.
                count_cols = dict()
                nobs_cols = dict()
                freq_cols = dict()
                cohorts_iterator = self._progress(
                    df_cohorts.itertuples(), desc="Compute karyotype frequencies"
                )
                for cohort in cohorts_iterator:
                    cohort_key = cohort.taxon, cohort.area, cohort.period
                    cohort_key_str = (
                        cohort.taxon + "_" + cohort.area + "_" + str(cohort.period)
                    )
                    n_samples = cohort.size
                    assert n_samples >= min_cohort_size
                    sample_indices = group_samples_by_cohort.indices[cohort_key]
                    kar_loc = kar_df.loc[sample_indices]

                    count_cols[f"count_{cohort_key_str}"] = [
                        len(kar_loc.query(f"karyotype_{inversion} == {i}"))
                        for i in range(0, 3)
                    ]
                    freq_cols[f"frq_{cohort_key_str}"] = [
                        c / n_samples for c in count_cols[f"count_{cohort_key_str}"]
                    ]
                    nobs = 2 * n_samples
                    nobs_cols[f"nobs_{cohort_key_str}"] = [nobs] * 3

                # Build a dataframe with the frequency columns.
                df_freqs = pd.DataFrame(freq_cols)
                df_counts = pd.DataFrame(count_cols)
                df_nobs = pd.DataFrame(nobs_cols)

                # Build the output dataset.
                ds_tmp = xr.Dataset()

                # Cohort variables.
                for coh_col in df_cohorts.columns:
                    ds_tmp[f"cohort_{coh_col}"] = "cohorts", df_cohorts[coh_col]

                # Variant labels
                ds_tmp["variant_label"] = (
                    "variants",
                    [
                        f"{inversion}_{allele}"
                        for allele in ["hom_ref", "het", "hom_alt"]
                    ],
                )

                # Event variables.
                ds_tmp["event_frequency"] = (
                    ("variants", "cohorts"),
                    df_freqs.to_numpy(),
                )
                ds_tmp["event_count"] = (
                    ("variants", "cohorts"),
                    df_counts.to_numpy(),
                )
                ds_tmp["event_nobs"] = (
                    ("variants", "cohorts"),
                    df_nobs.to_numpy(),
                )

                # Add confidence intervals.
                _add_frequency_ci(ds=ds_tmp, ci_method=ci_method)

                ds_list.append(ds_tmp)

            ds_out = xr.concat(ds_list, dim="variants", data_vars="minimal")

            return ds_out

        else:
            # Load sample metadata.
            df_samples = self.sample_metadata(
                sample_sets=sample_sets,
                sample_query=sample_query,
                sample_query_options=sample_query_options,
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

            # Access karyotypes.
            kar_df = self.karyotype(
                inversion=inversions,
                sample_sets=sample_sets,
                sample_query=sample_query,
                sample_query_options=sample_query_options,
            )

            # Count alleles.
            count_cols = dict()
            nobs_cols = dict()
            freq_cols = dict()
            cohorts_iterator = self._progress(
                df_cohorts.itertuples(), desc="Compute karyotype frequencies"
            )
            for cohort in cohorts_iterator:
                cohort_key = cohort.taxon, cohort.area, cohort.period
                cohort_key_str = (
                    cohort.taxon + "_" + cohort.area + "_" + str(cohort.period)
                )
                n_samples = cohort.size
                assert n_samples >= min_cohort_size
                sample_indices = group_samples_by_cohort.indices[cohort_key]
                kar_loc = kar_df.loc[sample_indices]

                count_cols[f"count_{cohort_key_str}"] = [
                    len(kar_loc.query(f"karyotype_{inversions} == {i}"))
                    for i in range(0, 3)
                ]
                freq_cols[f"frq_{cohort_key_str}"] = [
                    c / n_samples for c in count_cols[f"count_{cohort_key_str}"]
                ]
                nobs = 2 * n_samples
                nobs_cols[f"nobs_{cohort_key_str}"] = [nobs] * 3

            # Build a dataframe with the frequency columns.
            df_freqs = pd.DataFrame(freq_cols)
            df_counts = pd.DataFrame(count_cols)
            df_nobs = pd.DataFrame(nobs_cols)

            # Build the output dataset.
            ds_out = xr.Dataset()

            # Cohort variables.
            for coh_col in df_cohorts.columns:
                ds_out[f"cohort_{coh_col}"] = "cohorts", df_cohorts[coh_col]

            # Variant labels
            ds_out["variant_label"] = (
                "variants",
                [f"{inversions}_{allele}" for allele in ["hom_ref", "het", "hom_alt"]],
            )

            # Event variables.
            ds_out["event_frequency"] = (("variants", "cohorts"), df_freqs.to_numpy())
            ds_out["event_count"] = (
                ("variants", "cohorts"),
                df_counts.to_numpy(),
            )
            ds_out["event_nobs"] = (
                ("variants", "cohorts"),
                df_nobs.to_numpy(),
            )

            # Add confidence intervals.
            _add_frequency_ci(ds=ds_out, ci_method=ci_method)

            return ds_out
