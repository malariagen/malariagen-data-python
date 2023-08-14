"""Parameters for functions related to heterozygosity and runs of homozygosity."""

from typing import Tuple

import pandas as pd
from typing_extensions import Annotated, TypeAlias

window_size: TypeAlias = Annotated[
    int,
    "Number of sites per window.",
]

window_size_default: window_size = 20_000

phet_roh: TypeAlias = Annotated[
    float,
    "Probability of observing a heterozygote in a ROH.",
]

phet_roh_default: phet_roh = 0.001

phet_nonroh: TypeAlias = Annotated[
    Tuple[float, ...],
    "One or more probabilities of observing a heterozygote outside a ROH.",
]

phet_nonroh_default: phet_nonroh = (0.003, 0.01)

transition: TypeAlias = Annotated[
    float,
    """
    Probability of moving between states. A larger window size may call
    for a larger transitional probability.
    """,
]

transition_default: transition = 0.001

y_max: TypeAlias = Annotated[
    float,
    "Y axis limit.",
]

y_max_default: y_max = 0.03

df_roh: TypeAlias = Annotated[
    pd.DataFrame,
    """
    A DataFrame where each row provides data about a single run of
    homozygosity.
    """,
]

heterozygosity_height: TypeAlias = Annotated[
    int,
    "Height in pixels (px) of heterozygosity track.",
]

roh_height: TypeAlias = Annotated[
    int,
    "Height in pixels (px) of runs of homozygosity track.",
]
