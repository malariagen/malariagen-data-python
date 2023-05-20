"""Parameters common to functions accessing haplotype data."""

from typing_extensions import Annotated, TypeAlias

analysis: TypeAlias = Annotated[
    str,
    """
    Which haplotype phasing analysis to use. See the
    `phasing_analysis_ids` property for available values.
    """,
]
