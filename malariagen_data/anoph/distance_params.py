from typing import Literal

from typing_extensions import Annotated, TypeAlias

distance_metric: TypeAlias = Annotated[
    Literal[
        "cityblock",
        "euclidean",
        "sqeuclidean",
    ],
    "The metric to compute distance between genotypes in two samples.",
]

default_distance_metric: distance_metric = "cityblock"

nj_algorithm: TypeAlias = Annotated[
    Literal["dynamic", "rapid", "canonical"],
    "Neighbour-joining algorithm to use. The 'dynamic' algorithm is fastest.",
]

default_nj_algorithm: nj_algorithm = "dynamic"
