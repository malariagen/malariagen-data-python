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
    Literal["rapid", "canonical"],
    "Neighbour-joining algorithm to use.",
]

default_nj_algorithm: nj_algorithm = "rapid"
