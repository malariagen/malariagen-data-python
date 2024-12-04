"""Parameters for PCA functions."""

import numpy as np
import pandas as pd
from typing_extensions import Annotated, TypeAlias
from . import base_params

n_components: TypeAlias = Annotated[
    int,
    "Number of components to return.",
]

n_components_default: n_components = 20

df_pca: TypeAlias = Annotated[
    pd.DataFrame,
    """
    A dataframe of projections along principal components, one row per sample. The columns are:
        `sample_id` is the identifier of the sample,
        `partner_sample_id` is the identifier of the sample used by the partners who contributed it,
        `contributor` is the partner who contributed the sample,
        `country` is the country the sample was collected in,
        `location` is the location the sample was collected in,
        `year` is the year the sample was collected,
        `month` is the month the sample was collected,
        `latitude` is the latitude of the location the sample was collected in,
        `longitude` is the longitude of the location the sample was collected in,
        `sex_call` is the sex of the sample,
        `sample_set` is the sample set containing the sample,
        `release` is the release containing the sample,
        `quarter` is the quarter of the year the sample was collected,
        `study_id* is the identifier of the study the sample set containing the sample came from,
        `study_url` is the URL of the study the sample set containing the sample came from,
        `terms_of_use_expiry_date` is the date the terms of use for the sample expire,
        `terms_of_use_url` is the URL of the terms of use for the sample,
        `unrestricted_use` indicates whether the sample can be used without restrictions (e.g., if the terms of use of expired),
        `mean_cov` is mean value of the coverage,
        `median_cov` is the median value of the coverage,
        `modal_cov` is the mode of the coverage,
        `mean_cov_2L` is mean value of the coverage on 2L,
        `median_cov_2L` is the median value of the coverage on 2L,
        `mode_cov_2L` is the mode of the coverage on 2L,
        `mean_cov_2R` is mean value of the coverage on 2R,
        `median_cov_2R` is the median value of the coverage on 2R,
        `mode_cov_2R` is the mode of the coverage on 2R,
        `mean_cov_3L` is mean value of the coverage on 3L,
        `median_cov_3L` is the median value of the coverage on 3L,
        `mode_cov_3L` is the mode of the coverage on 3L,
        `mean_cov_3R` is mean value of the coverage on 3R,
        `median_cov_3R` is the median value of the coverage on 3R,
        `mode_cov_3R` is the mode of the coverage on 3R,
        `mean_cov_X` is mean value of the coverage on X,
        `median_cov_X` is the median value of the coverage on X,
        `mode_cov_X` is the mode of the coverage on X,
        `frac_gen_cov` is the faction of the genome covered,
        `divergence` is the divergence,
        `contam_pct` is the percentage of contamination,
        `contam_LLR` is the log-likelihood ratio of contamination,
        `aim_species_fraction_arab` is the fraction of the gambcolu vs. arabiensis AIMs that indicated arabiensis (this column is only present for *Ag3*),
        `aim_species_fraction_colu` is the fraction of the gambiae vs. coluzzii AIMs that indicated coluzzii (this column is only present for *Ag3*),
        `aim_species_fraction_colu_no2l` is the fraction of the gambiae vs. coluzzii AIMs that indicated coluzzii, not including the chromosome arm 2L which contains an introgression (this column is only present for *Ag3*),
        `aim_species_gambcolu_arabiensis` is the taxonomic group assigned by the gambcolu vs. arabiensis AIMs (this column is only present for *Ag3*),
        `aim_species_gambiae_coluzzi` is the taxonomic group assigned by the gambiae vs. coluzzii AIMs (this column is only present for *Ag3*),
        `aim_species_gambcolu_arabiensis` is the taxonomic group assigned by the combination of both AIMs analyses (this column is only present for *Ag3*),
        `country_iso` is the ISO code of the country the sample was collected in,
        `admin1_name` is the name of the first administrative level the sample was collected in,
        `admin1_iso` is the ISO code of the first administrative level the sample was collected in,
        `admin2_name` is the name of the second administrative level the sample was collected in,
        `taxon` is the taxon assigned to the sample by the combination of the AIMs analysis and the cohort analysis,
        `cohort_admin1_year` is the cohort the sample belongs to when samples are grouped by first administrative level and year,
        `cohort_admin1_month` is the cohort the sample belongs to when samples are grouped by first administrative level and month,
        `cohort_admin1_quarter` is the cohort the sample belongs to when samples are grouped by first administrative level and quarter,
        `cohort_admin2_year` is the cohort the sample belongs to when samples are grouped by second administrative level and year,
        `cohort_admin2_month` is the cohort the sample belongs to when samples are grouped by second administrative level and month,
        `cohort_admin2_quarter` is the cohort the sample belong to when samples are grouped by second administrative level and quarter.
        `PC?` is the projection along principal component ? (? being an integer between 1 and the number of components). There are as many such columns as components,
        `pca_fit` is whether this sample was used for fitting.
    """,
]

evr: TypeAlias = Annotated[
    np.ndarray,
    "An array of explained variance ratios, one per component.",
]

min_minor_ac_default: base_params.min_minor_ac = 2

max_missing_an_default: base_params.max_missing_an = 0
