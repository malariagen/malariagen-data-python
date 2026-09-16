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

distance_metric_default: Literal["hamming", "dxy"] = "hamming"

cohort_col: TypeAlias = Annotated[
    str,
    """
    Column name in sample metadata used to define cohorts for grouping,
    e.g., 'country', 'taxon', 'aim_species'.
    """,
]

cluster_method: TypeAlias = Annotated[
    Literal["cut", "burst"],
    """
    How to form flat clusters from the dendrogram. 'cut' cuts it at
    `cluster_threshold`. 'burst' instead selects clades whose internal diversity
    is far below the neutral expectation for the window, which needs no height
    and behaves the same at any sample size.
    """,
]

cluster_method_default: Literal["cut", "burst"] = "cut"

cluster_background: TypeAlias = Annotated[
    float,
    """
    Neutral expected pairwise distance for the window, in SNPs, used by the
    'burst' cluster method. If not given it is estimated from the data, which
    assumes some unswept haplotypes are present; supply a value where the region
    is swept throughout.
    """,
]
