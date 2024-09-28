from typing import Optional  # , Dict, Union, Callable, List

import pandas as pd
import numpy as np
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
            Compute Shaplotype frequencies for a region.
        """,
        returns="""
            A dataframe of haplotype frequencies, one row per haplotype.
        """,
        notes="""
            Cohorts with fewer samples than `min_cohort_size` will be excluded from
            output data frame.
        """,
    )
    def hap_frequencies(
        self,
        region: base_params.region,
        cohorts: base_params.cohorts,
        sample_query: Optional[base_params.sample_query] = None,
        min_cohort_size: base_params.min_cohort_size = 10,
        site_mask: Optional[base_params.site_mask] = None,
        sample_sets: Optional[base_params.sample_sets] = None,
        drop_invariant: frq_params.drop_invariant = True,
        effects: frq_params.effects = True,
        include_counts: frq_params.include_counts = False,
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
            site_mask=site_mask,
            sample_sets=sample_sets,
            sample_query=sample_query,
            chunks=chunks,
            inline_array=inline_array,
        )

        # Early check for no SNPs.
        if ds_hap.sizes["variants"] == 0:  # pragma: no cover
            raise ValueError("No SNPs available for the given region and site mask.")

        # Access genotypes.
        gt = ds_hap["call_genotype"].data
        with self._dask_progress(desc="Load SNP genotypes"):
            gt = gt.compute()

        # Count haplotypes.
        count_rows = dict()
        freq_rows = dict()
        freq_cols = dict()
        cohorts_iterator = self._progress(
            coh_dict.items(), desc="Compute allele frequencies"
        )
        for coh, loc_coh in cohorts_iterator:
            count_rows = {k: 0 for k in count_rows.keys()}
            n_samples = np.count_nonzero(loc_coh)
            assert n_samples >= min_cohort_size
            gt_coh = np.compress(loc_coh, gt, axis=1)
            for i in range(0, n_samples):
                for j in range(0, 2):
                    hap_hash = str(sha1(gt_coh[:, i, j].compute()).digest())
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
