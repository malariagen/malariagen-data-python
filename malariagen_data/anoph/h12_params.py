"""Parameter definitions for H12 analysis functions."""

from typing import Optional, Sequence, Union

from typing_extensions import Annotated, TypeAlias

from . import base_params

window_sizes: TypeAlias = Annotated[
    Sequence[int],
    """
    The sizes of windows (number of SNPs) used to calculate statistics within.
    """,
]

window_sizes_default: window_sizes = (100, 200, 500, 1000, 2000, 5000, 10000, 20000)

window_size: TypeAlias = Annotated[
    int,
    """
    The size of windows (number of SNPs) used to calculate statistics within.
    """,
]

multi_window_size: TypeAlias = Annotated[
    Union[window_size, dict[str, int]],
    """
    The size of windows (number of SNPs) used to calculate statistics within.
    """,
]

cohort_size_default: Optional[base_params.cohort_size] = None

min_cohort_size_default: base_params.min_cohort_size = 15

max_cohort_size_default: base_params.max_cohort_size = 50
