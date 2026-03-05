"""Parameters for Plink converter functions."""

from typing import Mapping, Union

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

output_name: TypeAlias = Annotated[
    str,
    """
    A custom name for the output PLINK files (without file extension).
    If not provided, a default name is generated from the analysis parameters.
    """,
]

phenotypes: TypeAlias = Annotated[
    Mapping[str, Union[int, float]],
    """
    A mapping of sample identifiers to phenotype values. In PLINK format,
    -9 indicates missing phenotype, 1 indicates control (unaffected),
    and 2 indicates case (affected). Continuous values can also be used.
    """,
]
