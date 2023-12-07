"""Parameters for PCA functions."""

import numpy as np
import pandas as pd
from typing_extensions import Annotated, TypeAlias

n_components: TypeAlias = Annotated[
    int,
    "Number of components to return.",
]

n_components_default: n_components = 20

df_pca: TypeAlias = Annotated[
    pd.DataFrame,
    """
    A dataframe of sample metadata, with columns "PC1", "PC2", "PC3",
    etc., added.
    """,
]

evr: TypeAlias = Annotated[
    np.ndarray,
    "An array of explained variance ratios, one per component.",
]
