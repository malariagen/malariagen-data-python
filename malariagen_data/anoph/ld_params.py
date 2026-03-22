"""Parameters for LD pruning functions."""

from typing_extensions import Annotated, TypeAlias

ld_window_size: TypeAlias = Annotated[
    int,
    "Window size in number of SNPs for LD pruning.",
]

ld_window_size_default: ld_window_size = 500

ld_window_step: TypeAlias = Annotated[
    int,
    "Step size in number of SNPs for LD pruning.",
]

ld_window_step_default: ld_window_step = 200

ld_threshold: TypeAlias = Annotated[
    float,
    "r-squared threshold for LD pruning. SNP pairs with r-squared above "
    "this threshold will be considered linked.",
]

ld_threshold_default: ld_threshold = 0.1
