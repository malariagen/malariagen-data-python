"""Parameters for haplotype clustering functions."""

from typing import Literal

from typing_extensions import Annotated, TypeAlias

linkage_method: TypeAlias = Annotated[
    Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"],
    """
    The linkage algorithm to use. See the Linkage Methods section of the
    scipy.cluster.hierarchy.linkage docs for full descriptions.
    """,
]

linkage_method_default: linkage_method = "single"

count_sort: TypeAlias = Annotated[
    bool,
    """
    For each node n, the order (visually, from left-to-right) n's two descendant
    links are plotted is determined by this parameter. If True, the child with
    the minimum number of original objects in its cluster is plotted first. Note
    distance_sort and count_sort cannot both be True.
    """,
]

distance_sort: TypeAlias = Annotated[
    bool,
    """
    For each node n, the order (visually, from left-to-right) n's two descendant
    links are plotted is determined by this parameter. If True, The child with the
    minimum distance between its direct descendants is plotted first.
    """,
]
