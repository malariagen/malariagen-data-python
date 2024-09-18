"""Parameter definitions for H12 analysis functions."""

from typing import Optional, Sequence

from typing_extensions import Annotated, TypeAlias

from . import base_params, gplt_params, hap_params

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

cohort_size_default: Optional[base_params.cohort_size] = None

min_cohort_size_default: base_params.min_cohort_size = 15

max_cohort_size_default: base_params.max_cohort_size = 50


class sample_query_params:
    def __init__(
        self,
        sample_query: base_params.sample_query,
        title: Optional[gplt_params.title],
        window_size: window_size,
        analysis: Optional[hap_params.analysis] = base_params.DEFAULT,
        cohort_size: Optional[base_params.cohort_size] = cohort_size_default,
        min_cohort_size: Optional[
            base_params.min_cohort_size
        ] = min_cohort_size_default,
        max_cohort_size: Optional[
            base_params.max_cohort_size
        ] = max_cohort_size_default,
    ) -> None:
        self.sample_query = sample_query
        if title:
            self.title = title
        else:
            self.title = sample_query
        window_size = window_size
        analysis = analysis
        cohort_size = cohort_size
        min_cohort_size = min_cohort_size
        max_cohort_size = max_cohort_size


sample_queries: TypeAlias = Annotated[
    Sequence[sample_query_params],
    """
Work in progress!
    """,
]
