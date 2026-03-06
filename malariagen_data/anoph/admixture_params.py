from typing import Annotated, TypeAlias

k_value: TypeAlias = Annotated[
    int,
    """
    The number of ancestral populations (K) to assume for
    ADMIXTURE analysis.
    """,
]

k_value_default: int = 3

max_samples_per_cohort: TypeAlias = Annotated[
    int,
    """
    Maximum number of samples to include per cohort for
    downsampling individuals.
    """,
]

max_snps: TypeAlias = Annotated[
    int,
    """
    Maximum total number of SNPs to retain after downsampling.
    """,
]
