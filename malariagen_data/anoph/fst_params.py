"""Parameter definitions for Fst functions."""

from typing import Optional, Union, Literal

import pandas as pd
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

df_pairwise_fst: TypeAlias = Annotated[
    pd.DataFrame,
    """
    A dataframe of pairwise Fst and standard error values.
    """,
]

annotate_se: TypeAlias = Annotated[
    Optional[Union[Literal["standard error", "Z score"], bool]],
    """
    How to annotate the upper-right corner of the plot. Default behaviour is using Fst, other options
    are using the standard error (if annotate_se is True or 'standard error') or the Z score of the two
    cohorts being the same (if annotate_se is 'Z score').
    """,
]
