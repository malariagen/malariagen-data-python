"""Parameters for VCF exporter functions."""

from typing import Optional, Tuple

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

vcf_max_region_size: TypeAlias = Annotated[
    Optional[int],
    """
    Maximum total size of `region` in base pairs. The VCF is loaded in
    full by IGV, so large regions are rejected. Set to None to disable
    this check.
    """,
]
