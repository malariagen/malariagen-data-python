"""Parameters for haplotype clustering functions."""

from .clustering_params import linkage_method
from typing_extensions import Annotated, TypeAlias, Literal

linkage_method_default: linkage_method = "single"

distance_metric: TypeAlias = Annotated[
    Literal["hamming", "dxy"],
    """
    The distance metric to use for calculating pairwise distances between haplotypes.
    'hamming' computes the Hamming distance (number of differing SNPs) between haplotypes.
    'dxy' computes the average number of nucleotide differences per site between haplotypes.
    """,
]

distance_metric_default: str = "hamming"
