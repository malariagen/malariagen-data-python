"""Parameters for VCF exporter functions."""

from typing_extensions import Annotated, TypeAlias

vcf_output_path: TypeAlias = Annotated[
    str,
    """
    Path to write the VCF output file. Use a `.vcf.gz` extension to enable
    gzip compression.
    """,
]
