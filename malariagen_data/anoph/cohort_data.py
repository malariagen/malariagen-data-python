from typing import Optional

import pandas as pd
from numpydoc_decorator import doc

from ..util import check_types
from . import base_params
from .base import AnophelesBase


class AnophelesCohortData(AnophelesBase):
    def __init__(
        self,
        cohorts_analysis: Optional[str] = None,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

        # If provided, this analysis version will override the
        # default value provided in the release configuration.
        self._cohorts_analysis_override = cohorts_analysis

    @property
    def _cohorts_analysis(self):
        if self._cohorts_analysis_override:
            return self._cohorts_analysis_override
        else:
            # N.B., this will return None if the key is not present in the
            # config.
            return self.config.get("DEFAULT_COHORTS_ANALYSIS")

    @check_types
    @doc(
        summary="""
            Read data for a specific cohort set, including cohort size,
            country code, taxon, administrative units name, ISO code, geoBoundaries
            shape ID and representative latitude and longitude points.
        """,
        parameters=dict(
            cohort_set="""
                A cohort set name. Accepted values are:
                "admin1_month", "admin1_quarter", "admin1_year",
                "admin2_month", "admin2_quarter", "admin2_year".
            """
        ),
        returns="A dataframe of cohort data, one row per cohort.",
    )
    def cohorts(
        self,
        cohort_set: base_params.cohorts,
    ) -> pd.DataFrame:
        major_version_path = self._major_version_path
        cohorts_analysis = self._cohorts_analysis

        path = f"{major_version_path[:2]}_cohorts/cohorts_{cohorts_analysis}/cohorts_{cohort_set}.csv"

        # Read the manifest into a pandas dataframe.
        with self.open_file(path) as f:
            df_cohorts = pd.read_csv(f, sep=",", na_values="")

        # Ensure all column names are lower case.
        df_cohorts.columns = [c.lower() for c in df_cohorts.columns]

        return df_cohorts
