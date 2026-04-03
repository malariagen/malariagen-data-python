from typing import Optional, Dict, Any
import numpy as np
import scipy.stats

from numpydoc_decorator import doc  # type: ignore

from . import base_params, phenotype_params
from .phenotypes import AnophelesPhenotypeData
from ..util import _check_types, Region


class AnophelesAssociationAnalysis(AnophelesPhenotypeData):
    """
    Provides methods for testing statistical associations between
    specific variants and phenotypic traits.
    Inherited by AnophelesDataResource subclasses (e.g., Ag3).
    """

    def __init__(self, **kwargs):
        # Cooperatively initialize parent classes
        super().__init__(**kwargs)

    @_check_types
    @doc(
        summary="Test for association between a specific variant and a binary phenotype.",
        parameters=dict(
            position="The 1-based coordinate of the variant.",
        ),
        returns=dict(
            stats="A dictionary containing the Fisher's Exact test odds ratio, p-value, and contingency table counts."
        ),
    )
    def variant_association(
        self,
        region: base_params.region,
        position: int,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        insecticide: Optional[phenotype_params.insecticide] = None,
        dose: Optional[phenotype_params.dose] = None,
        phenotype: Optional[phenotype_params.phenotype] = None,
        cohort_size: Optional[base_params.cohort_size] = None,
        min_cohort_size: Optional[base_params.min_cohort_size] = None,
        max_cohort_size: Optional[base_params.max_cohort_size] = None,
    ) -> Dict[str, Any]:
        """
        Extract phenotype and genotype data for a specific variant position and apply
        Fisher's Exact Test to determine the statistical association between possessing
        an alternate allele and the phenotype.
        """
        # Parse the region to ensure we only pull the exact variant coordinate
        # Fetching an entire chromosome of SNPs (e.g., '2L') would be extremely slow!
        r = Region(region)
        target_region = f"{r.contig}:{position}-{position}"

        # Fetch the merged multidimensional xarray for the target coordinate
        ds = self.phenotypes_with_snps(
            region=target_region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            cohort_size=cohort_size,
            min_cohort_size=min_cohort_size,
            max_cohort_size=max_cohort_size,
        )

        if "samples" not in ds.sizes or ds.sizes["samples"] == 0:
            raise ValueError("No matching records found for the given criteria.")

        # If insecticide/dose/phenotype filters were provided, apply them
        # (Alternatively, users can embed these in sample_query)
        valid_indices = np.ones(ds.sizes["samples"], dtype=bool)
        if insecticide is not None:
            valid_indices &= ds["insecticide"].values == insecticide
        if dose is not None:
            valid_indices &= ds["dose"].values == dose
        if phenotype is not None:
            valid_indices &= ds["phenotype"].values.astype(str) == str(phenotype)

        # Check if the variant_position exists in the extracted region
        var_positions = ds["variant_position"].values
        pos_mask = var_positions == position
        if not np.any(pos_mask):
            raise ValueError(
                f"Variant position {position} not found in region {region}."
            )

        # Sub-select data arrays
        phenos = ds["phenotype_binary"].values[valid_indices]
        # shape is (variants, samples, ploidy)
        # Select specifically the row for `position`
        gt = ds["call_genotype"].values[pos_mask][0]  # shape (samples, ploidy)
        gt = gt[valid_indices]

        # Ignore missing phenotypes (NaN) and missing calls (-1)
        valid_mask = ~np.isnan(phenos) & (gt.min(axis=1) >= 0)
        phenos_valid = phenos[valid_mask]
        gt_valid = gt[valid_mask]

        # Define 2x2 categorical buckets
        # "Has Alt": True if any allele in the genotype call is > 0 (e.g. 0/1 or 1/1)
        has_alt = (gt_valid > 0).any(axis=1)
        has_ref = ~has_alt  # (i.e. entirely 0/0)

        pheno_positive = phenos_valid == 1
        pheno_negative = phenos_valid == 0

        # Build Contingency Table:
        #           Alt     Ref
        # Pos        a       b
        # Neg        c       d
        a = np.sum(pheno_positive & has_alt)
        b = np.sum(pheno_positive & has_ref)
        c = np.sum(pheno_negative & has_alt)
        d = np.sum(pheno_negative & has_ref)

        table = [[a, b], [c, d]]
        res = scipy.stats.fisher_exact(table, alternative="two-sided")

        # In newer scipy versions (1.7+): res.statistic is OR, res.pvalue is P-val
        # Support older/newer scipy return tuples safely
        odds_ratio = res[0]
        p_value = res[1]

        return {
            "region": region,
            "position": position,
            "contingency_table": table,
            "phenotype_positive_alt": int(a),
            "phenotype_positive_ref": int(b),
            "phenotype_negative_alt": int(c),
            "phenotype_negative_ref": int(d),
            "odds_ratio": float(odds_ratio),
            "p_value": float(p_value),
            "total_valid_samples": int(len(phenos_valid)),
        }
