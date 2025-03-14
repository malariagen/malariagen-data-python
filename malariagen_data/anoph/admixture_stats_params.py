"""Parameter definitions for admixture functions."""

from typing import Optional
from typing_extensions import Annotated, TypeAlias

from . import base_params

cohort_size_default: Optional[base_params.cohort_size] = None
min_cohort_size_default: base_params.min_cohort_size = 10
max_cohort_size_default: base_params.max_cohort_size = 50

segregating_mode: TypeAlias = Annotated[
    str,
    """
    Define the way segregating variants are chosen. Available options are "recipient" or "all" to use
    sites segregating within both the recipient and source population cohorts.
    """,
]
