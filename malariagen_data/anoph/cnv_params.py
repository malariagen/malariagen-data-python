"""Parameters common to functions accessing CNV data."""

from typing import Optional

from typing_extensions import Annotated, TypeAlias

max_coverage_variance: TypeAlias = Annotated[
    Optional[float],
    """
    Remove samples if coverage variance exceeds this value.
    """,
]

max_coverage_variance_default: max_coverage_variance = 0.2

coverage_calls_analysis: TypeAlias = Annotated[
    str,
    """
    Which coverage calls analysis to use. See the
    `coverage_calls_analysis_ids` property for available values.
    """,
]
