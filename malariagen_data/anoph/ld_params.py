"""Parameter definitions for LD pruning functions."""

from typing_extensions import Annotated, TypeAlias

size: TypeAlias = Annotated[
    int,
    "The number of SNPs in each window used for LD pruning.",
]

size_default: size = 500

step: TypeAlias = Annotated[
    int,
    "The number of SNPs to advance the window by in each iteration.",
]

step_default: step = 200

threshold: TypeAlias = Annotated[
    float,
    "The r-squared threshold above which SNPs are considered to be in linkage disequilibrium.",
]

threshold_default: threshold = 0.1
