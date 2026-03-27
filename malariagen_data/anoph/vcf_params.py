"""Parameters for VCF converter functions."""

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

out: TypeAlias = Annotated[
    str,
    """
    A string specifying the output VCF file name (without extension). The VCF
    output file will be written as ``{output_dir}/{out}.vcf``. If not provided,
    a default name is generated from the SNP selection parameters.
    """,
]
