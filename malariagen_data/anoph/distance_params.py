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

center_x: TypeAlias = Annotated[int | float, "X coordinate where plotting is centered."]

center_y: TypeAlias = Annotated[int | float, "Y coordinate where plotting is centered."]

arc_start: TypeAlias = Annotated[int | float, "Angle where tree layout begins."]

arc_stop: TypeAlias = Annotated[int | float, "Angle where tree layout ends."]

edge_legend: TypeAlias = Annotated[
    bool, "Show legend entries for the different edge (line) colors."
]

leaf_legend: TypeAlias = Annotated[
    bool,
    "Show legend entries for the different leaf node (scatter) colors and symbols.",
]
