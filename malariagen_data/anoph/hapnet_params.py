"""Parameters for haplotype network functions."""

from typing import Mapping

from typing_extensions import Annotated, TypeAlias

max_dist: TypeAlias = Annotated[
    int,
    "Join network components up to a maximum distance of 2 SNP differences.",
]

max_dist_default: max_dist = 2

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
