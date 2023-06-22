"""Parameter definitions for XPEHH analysis functions."""

from typing import Tuple, Union

import numpy as np
from typing_extensions import Annotated, TypeAlias

from . import base_params

window_size: TypeAlias = Annotated[
    int,
    """
        The size of window in number of SNPs used to summarise XP-EHH over.
        If None, per-variant XP-EHH values are returned.
        """,
]
window_size_default: window_size = 200
min_cohort_size_default: base_params.min_cohort_size = 15
max_cohort_size_default: base_params.max_cohort_size = 50
percentiles: TypeAlias = Annotated[
    Union[int, Tuple[int, ...]],
    """
    If window size is specified, this returns the XP-EHH percentiles
    for each window.
    """,
]
percentiles_default: percentiles = (50, 75, 100)
filter_min_maf: TypeAlias = Annotated[
    float,
    """
    Minimum minor allele frequency to use for filtering prior to passing
    haplotypes to allel.xpehh function
    """,
]
filter_min_maf_default: filter_min_maf = 0.05
map_pos: TypeAlias = Annotated[
    np.ndarray,
    """
    Variant positions (genetic map distance).
    """,
]
min_ehh: TypeAlias = Annotated[
    float,
    """
    Minimum EHH beyond which to truncate integrated haplotype homozygosity
    calculation.
    """,
]
min_ehh_default: min_ehh = 0.05
max_gap: TypeAlias = Annotated[
    int,
    """
    Do not report scores if EHH spans a gap larger than this number of
    base pairs.
    """,
]
max_gap_default: max_gap = 200_000
gap_scale: TypeAlias = Annotated[
    int, "Rescale distance between variants if gap is larger than this value."
]
gap_scale_default: gap_scale = 20_000
include_edges: TypeAlias = Annotated[
    bool,
    """
    If True, report scores even if EHH does not decay below min_ehh at the
    end of the chromosome.
    """,
]
use_threads: TypeAlias = Annotated[
    bool, "If True, use multiple threads to compute XP-EHH."
]
palette: TypeAlias = Annotated[
    str, "Name of bokeh palette to use for plotting multiple percentiles."
]
palette_default: palette = "Blues"
