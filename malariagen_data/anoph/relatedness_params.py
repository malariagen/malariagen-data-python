"""Parameters for relatedness functions."""

from typing_extensions import Annotated, TypeAlias

maf: TypeAlias = Annotated[
    float,
    "Individual minor allele frequency filter. If an individual's estimated "
    "individual-specific minor allele frequency at a SNP is less than this value, "
    "that SNP will be excluded from the analysis for that individual. "
    "Must be between (0.0, 0.1).",
]

maf_default: maf = 0.01
