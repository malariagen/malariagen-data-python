"""Parameters for Plink converter functions."""

from typing_extensions import Annotated, TypeAlias

overwrite: TypeAlias = Annotated[
    bool,
    """
    A boolean indicating whether a previously written file with the same name ought
    to be overwritten. Default is False.
    """,
]

output_dir: TypeAlias = Annotated[
    str,
    """
    A string indicating the desired output file location.
    """,
]
