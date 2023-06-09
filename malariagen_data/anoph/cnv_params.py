"""Parameters common to functions accessing CNV data."""

from typing import Optional, Union

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

y_max: TypeAlias = Annotated[
    Union[float, str],
    "Y axis limit or 'auto'.",
]

y_max_default: y_max = "auto"
