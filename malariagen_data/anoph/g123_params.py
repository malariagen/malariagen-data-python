"""Parameter definitions for G123 analysis functions."""

from typing import Sequence

from typing_extensions import Annotated, TypeAlias

from . import base_params

sites: TypeAlias = Annotated[
    str,
    """
    Which sites to use: 'all' includes all sites that pass
    site filters; 'segregating' includes only segregating sites for
    the given cohort; or a phasing analysis identifier can be
    provided to use sites from the haplotype data, which is an
    approximation to finding segregating sites in the entire Ag3.0
    (gambiae complex) or Af1.0 (funestus) cohort.
    """,
]

window_sizes: TypeAlias = Annotated[
    Sequence[int],
    """
    The sizes of windows (number of sites) used to calculate statistics within.
    """,
]

window_sizes_default: window_sizes = (100, 200, 500, 1000, 2000, 5000)

window_size: TypeAlias = Annotated[
    int,
    """
    The size of windows (number of sites) used to calculate statistics within.
    """,
]

min_cohort_size_default: base_params.min_cohort_size = 20

max_cohort_size_default: base_params.max_cohort_size = 50
