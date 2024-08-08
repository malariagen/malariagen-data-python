"""Parameters for hierarchical clustering functions."""

from typing import Literal

from typing_extensions import Annotated, TypeAlias

linkage_method: TypeAlias = Annotated[
    Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"],
    """
    The linkage algorithm to use. See the Linkage Methods section of the
    scipy.cluster.hierarchy.linkage docs for full descriptions.
    """,
]

leaf_y: TypeAlias = Annotated[
    int,
    "Y coordinate at which to plot the leaf markers.",
]
