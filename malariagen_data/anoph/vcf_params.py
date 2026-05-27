"""Parameters for VCF exporter functions."""

from typing import Tuple

from typing_extensions import Annotated, TypeAlias

vcf_output_path: TypeAlias = Annotated[
    str,
    """
    Path to write the VCF output file. Use a `.vcf.gz` extension to enable
    gzip compression.
    """,
]

vcf_fields: TypeAlias = Annotated[
    Tuple[str, ...],
    """
    FORMAT fields to include in the VCF output. Must include "GT".
    Supported fields: "GT", "GQ", "AD", "MQ".
    """,
]
