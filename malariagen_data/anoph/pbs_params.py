"""Parameter definitions for PBS functions."""

from typing import Optional

from typing_extensions import Annotated, TypeAlias

from . import base_params

# N.B., window size can mean different things for different functions
window_size: TypeAlias = Annotated[
    int,
    "The size of windows (number of sites) used to calculate statistics within.",
]

cohort_size_default: Optional[base_params.cohort_size] = None
min_cohort_size_default: base_params.min_cohort_size = 15
max_cohort_size_default: base_params.max_cohort_size = 50

normed: TypeAlias = Annotated[
    bool,
    """
    If True, normalise the PBS values by the sum of the divergence times.
    This can help to identify extreme outlier loci. Default is True.
    """,
]
normed_default: bool = True

min_snps_threshold: TypeAlias = Annotated[
    int,
    """
    Minimum number of SNP sites required for the PBS GWSS computation. If
    fewer sites are available, a ValueError is raised.
    """,
]

window_adjustment_factor: TypeAlias = Annotated[
    int,
    """
    If window_size is >= the number of available SNP sites, the window_size
    is automatically adjusted to number_of_snps // window_adjustment_factor.
    """,
]
