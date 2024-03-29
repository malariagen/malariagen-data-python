"""Parameters for diplotype clustering functions."""

from typing import Literal

from typing_extensions import Annotated, TypeAlias

linkage_method: TypeAlias = Annotated[
    Literal["single", "complete", "average", "weighted", "centroid", "median", "ward"],
    """
    The linkage algorithm to use. See the Linkage Methods section of the
    scipy.cluster.hierarchy.linkage docs for full descriptions.
    """,
]

linkage_method_default: linkage_method = "complete"

distance_metric: TypeAlias = Annotated[
    Literal["cityblock", "euclidean"],
    """
    The distance metric to use. Either "cityblock" or "euclidean".
    """,
]

distance_metric_default: distance_metric = "cityblock"
