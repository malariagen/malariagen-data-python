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

nj_algorithm: TypeAlias = Annotated[
    Literal["rapid", "canonical"],
    "Neighbour-joining algorithm to use.",
]
