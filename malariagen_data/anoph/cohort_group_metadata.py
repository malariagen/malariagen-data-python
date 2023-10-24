from typing import Optional

import pandas as pd
from numpydoc_decorator import doc

from ..util import check_types
from . import base_params
from .base import AnophelesBase


class AnophelesCohortGroupMetadata(AnophelesBase):
    def __init__(
        self,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

    @check_types
    @doc(
        summary="""
            Read metadata for a specific cohort group, including cohort size,
            country code, taxon, administrative units name, ISO code, geoBoundaries
            shape ID and representative latitude and longitude points.
        """,
        parameters=dict(
            cohort_group="""
                A cohort group name. Accepted values are:
                "admin1_month", "admin1_quarter", "admin1_year",
                "admin2_month", "admin2_quarter", "admin2_year".
            """
        ),
        returns="A dataframe of cohort metadata, one row per cohort.",
    )
    def cohort_group_metadata(
        self,
        cohort_group: base_params.cohorts,
        cohort_group_query: Optional[base_params.cohort_group_query] = None,
    ) -> pd.DataFrame:
        major_version_path = self._major_version_path
        cohorts_analysis = self.config.get("DEFAULT_COHORTS_ANALYSIS")

        path = f"{major_version_path[:2]}_cohorts/cohorts_{cohorts_analysis}/cohorts_{cohort_group}.csv"

        # Read the manifest into a pandas dataframe.
        with self.open_file(path) as f:
            df_cohorts = pd.read_csv(f, sep=",", na_values="")

        # Ensure all column names are lower case.
        df_cohorts.columns = [c.lower() for c in df_cohorts.columns]

        # Apply a cohort group selection.
        if cohort_group_query is not None:
            # Assume a pandas query string.
            df_cohorts = df_cohorts.query(cohort_group_query)
            df_cohorts = df_cohorts.reset_index(drop=True)

        return df_cohorts.copy()
