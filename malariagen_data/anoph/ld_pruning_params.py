from typing import Annotated, TypeAlias

r2_threshold: TypeAlias = Annotated[
    float,
    """
    The r² threshold for LD pruning. SNP pairs with r² above this
    threshold are considered to be in linkage disequilibrium,
    and one of the pair will be removed.
    """,
]

r2_threshold_default: float = 0.1

window_size: TypeAlias = Annotated[
    int,
    """
    The number of SNPs to consider in each sliding window
    when computing pairwise LD.
    """,
]

window_size_default: int = 500

window_step: TypeAlias = Annotated[
    int,
    """
    The number of SNPs to advance the window by after each iteration.
    """,
]

window_step_default: int = 200

n_iter: TypeAlias = Annotated[
    int,
    """
    Number of iterations of LD pruning to perform. Each iteration
    may remove additional SNPs that become linked after previous
    removals.
    """,
]

n_iter_default: int = 5
