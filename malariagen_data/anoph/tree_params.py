"""Parameters for haplotype clustering functions."""

from typing_extensions import Annotated, TypeAlias

count_sort: TypeAlias = Annotated[
    bool,
    """
    If True, for each node n, the child with the minimum number of descendants is
    plotted first. Note distance_sort and count_sort cannot both be True.
    """,
]

distance_sort: TypeAlias = Annotated[
    bool,
    """
    If True, for each node n, if True, the child with the minimum distance between
    is plotted first. Note distance_sort and count_sort cannot both be True.
    """,
]
