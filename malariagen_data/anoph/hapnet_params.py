"""Parameters for haplotype network functions."""

from typing import List, Mapping

from typing_extensions import Annotated, TypeAlias

max_dist: TypeAlias = Annotated[
    int,
    "Join network components up to a maximum distance of 2 SNP differences.",
]

max_dist_default: max_dist = 2

color: TypeAlias = Annotated[
    str,
    """
    Identifies a column in the sample metadata which determines the colour
    of pie chart segments within nodes.
    """,
]

color_discrete_sequence: TypeAlias = Annotated[
    List, "Provide a list of colours to use."
]

color_discrete_map: TypeAlias = Annotated[
    Mapping, "Provide an explicit mapping from values to colours."
]

category_order: TypeAlias = Annotated[
    List,
    "Control the order in which values appear in the legend.",
]

node_size_factor: TypeAlias = Annotated[
    int,
    "Control the sizing of nodes.",
]

node_size_factor_default: node_size_factor = 50

layout: TypeAlias = Annotated[
    str,
    "Name of the network layout to use to position nodes.",
]

layout_default: layout = "cose"

layout_params: TypeAlias = Annotated[
    Mapping,
    "Additional parameters to the layout algorithm.",
]
