from typing import Tuple, Optional

import allel
import numpy as np
import numba

from . import base_params
from . import admixture_stats_params
from .snp_data import AnophelesSampleMetadata, AnophelesSnpData


@numba.njit
def _remap_observed(ac_full):
    """Remap allele indices to remove unobserved alleles."""
    n_variants = ac_full.shape[0]
    n_alleles = ac_full.shape[1]
    # Create the output array - this is an allele mapping array,
    # that specifies how to recode allele indices.
    mapping = np.empty((n_variants, n_alleles), dtype=np.int32)
    mapping[:] = -1
    # Iterate over variants.
    for i in range(n_variants):
        # This will be the new index that we are mapping this allele to, if the
        # allele count is not zero.
        j_out = 0
        # Iterate over columns (alleles) in the input array.
        for j in range(n_alleles):
            # Access the count for the jth allele.
            c = ac_full[i, j]
            if c > 0:
                # We have found a non-zero allele count, remap the allele.
                mapping[i, j] = j_out
                j_out += 1
    return mapping


class AnophelesAdmixtureAnalysis(
    AnophelesSnpData,
    AnophelesSampleMetadata,
):
    def __init__(
        self,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

    def patterson_f3(
        self,
        recipient_query: base_params.sample_query,
        source1_query: base_params.sample_query,
        source2_query: base_params.sample_query,
        region: base_params.region,
        site_mask: Optional[base_params.site_mask] = base_params.DEFAULT,
        n_jack: base_params.n_jack = 200,
        max_missing_an: base_params.max_missing_an = 0,
        segregating_mode: admixture_stats_params.segregating_mode = "recipient",
        cohort_size: Optional[
            base_params.cohort_size
        ] = admixture_stats_params.cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = admixture_stats_params.min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = admixture_stats_params.max_cohort_size_default,
    ) -> Tuple[float, float, float]:
        # Purely for conciseness, internally here we use the scikit-allel convention
        # of labelling the recipient population "C" and the source populations as
        # "A" and "B".

        # Compute allele counts for the three cohorts.
        acc = self.snp_allele_counts(
            sample_query=recipient_query,
            region=region,
            site_mask=site_mask,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
        )
        aca = self.snp_allele_counts(
            sample_query=source1_query,
            region=region,
            site_mask=site_mask,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
        )
        acb = self.snp_allele_counts(
            sample_query=source2_query,
            region=region,
            site_mask=site_mask,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
        )

        # Locate biallelic variants and deal with missingness.
        ac = acc + aca + acb
        loc_bi = allel.AlleleCountsArray(ac).is_biallelic()
        anc = np.sum(acc, axis=1)
        ana = np.sum(aca, axis=1)
        anb = np.sum(acb, axis=1)
        an = np.sum(ac, axis=1)  # no. required for sample size
        n_chroms = an.max()
        an_missing = n_chroms - an
        # In addition to applying the max_missing_an threshold, also make sure that
        # all three cohorts have nonzero allele counts.
        loc_nomiss = (an_missing <= max_missing_an) & (anc > 0) & (ana > 0) & (anb > 0)
        loc_sites = loc_bi & loc_nomiss
        acc_bi = acc[loc_sites]
        aca_bi = aca[loc_sites]
        acb_bi = acb[loc_sites]
        ac_bi = ac[loc_sites]

        # Squeeze the allele counts so we end up with only two columns.
        sqz_mapping = _remap_observed(ac_bi)
        acc_sqz = allel.AlleleCountsArray(acc_bi).map_alleles(sqz_mapping, max_allele=1)
        aca_sqz = allel.AlleleCountsArray(aca_bi).map_alleles(sqz_mapping, max_allele=1)
        acb_sqz = allel.AlleleCountsArray(acb_bi).map_alleles(sqz_mapping, max_allele=1)

        # Keep only segregating variants.
        if segregating_mode == "recipient":
            loc_seg = acc_sqz.is_segregating()
        elif segregating_mode == "all":
            loc_seg = (
                acc_sqz.is_segregating()
                & aca_sqz.is_segregating()
                & acb_sqz.is_segregating()
            )
        else:
            raise ValueError("Invalid value for 'segregating_mode' parameter.")
        if loc_seg.sum() == 0:
            raise ValueError("No segregating variants found")
        else:
            print(f"Using {loc_seg.sum()} segregating variants for F3 calculation.")
        acc_seg = acc_sqz[loc_seg]
        aca_seg = aca_sqz[loc_seg]
        acb_seg = acb_sqz[loc_seg]

        # Compute f3 statistic.
        blen = acc_seg.shape[0] // n_jack
        f3, se, z, _, _ = allel.average_patterson_f3(
            acc_seg,
            aca_seg,
            acb_seg,
            blen=blen,
            normed=True,
        )

        return f3, se, z
