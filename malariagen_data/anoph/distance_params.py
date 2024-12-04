from typing import Literal

from typing_extensions import Annotated, TypeAlias

import numpy as np

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

dist: TypeAlias = Annotated[
    np.ndarray,
    """
    A numpy array containing the distance between each pair of samples.
    """,
]

Z: TypeAlias = Annotated[
    np.ndarray,
    """
    A neighbour-joining tree encoded as a numpy array. Each row in the
    array contains data for one internal node in the tree, in the order
    in which they were created by the neighbour-joining algorithm.
    Within each row there are five values: left child node identifier,
    right child node identifier, distance to left child, distance to
    right child, total number of leaves. This data structure is similar
    to that returned by scipy's hierarchical clustering functions,
    except that here we have two distance values for each internal node
    rather than one because distances to the children may be different.
    """,
]

samples: TypeAlias = Annotated[np.ndarray, "The list of the sample identifiers"]

n_snps_used: TypeAlias = Annotated[int, "The number of SNPs used"]

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
