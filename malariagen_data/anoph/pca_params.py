"""Parameters for PCA functions."""

import numpy as np
import pandas as pd
from typing_extensions import Annotated, TypeAlias

n_snps: TypeAlias = Annotated[
    int,
    """
    The desired number of SNPs to use when running the analysis.
    SNPs will be evenly thinned to approximately this number.
    """,
]

thin_offset: TypeAlias = Annotated[
    int,
    """
    Starting index for SNP thinning. Change this to repeat the analysis
    using a different set of SNPs.
    """,
]

thin_offset_default: thin_offset = 0

min_minor_ac: TypeAlias = Annotated[
    int,
    """
    The minimum minor allele count. SNPs with a minor allele count
    below this value will be excluded prior to thinning.
    """,
]

min_minor_ac_default: min_minor_ac = 2

max_missing_an: TypeAlias = Annotated[
    int,
    """
    The maximum number of missing allele calls to accept. SNPs with
    more than this value will be excluded prior to thinning. Set to 0
    (default) to require no missing calls.
    """,
]

max_missing_an_default = 0

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
