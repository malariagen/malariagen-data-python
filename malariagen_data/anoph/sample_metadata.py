import io
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd
from numpydoc_decorator import doc

from .base import AnophelesBase, base_params


class AnophelesSampleMetadata(AnophelesBase):
    def __init__(
        self,
        cohorts_analysis: Optional[str] = None,
        aim_analysis: Optional[str] = None,
        aim_metadata_columns: Optional[Sequence[str]] = None,
        aim_metadata_dtype: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

        # If provided, this analysis version will override the
        # default value provided in the release configuration.
        self._cohorts_analysis_override = cohorts_analysis

        # If provided, this analysis version will override the
        # default value provided in the release configuration.
        self._aim_analysis_override = aim_analysis

        # N.B., the expected AIM metadata columns may vary between
        # data resources, and so column names and dtype need to be
        # passed in as parameters.
        self._aim_metadata_columns = aim_metadata_columns
        self._aim_metadata_dtype: Dict[str, Any] = dict()
        if isinstance(aim_metadata_dtype, Mapping):
            self._aim_metadata_dtype.update(aim_metadata_dtype)
        self._aim_metadata_dtype["sample_id"] = object

        # Initialize cache attributes.
        # TODO

    def _general_metadata_paths(self, *, sample_sets: List[str]) -> Dict[str, str]:
        paths = dict()
        for sample_set in sample_sets:
            release = self.lookup_release(sample_set=sample_set)
            release_path = self._release_to_path(release=release)
            path = f"{release_path}/metadata/general/{sample_set}/samples.meta.csv"
            paths[sample_set] = path
        return paths

    def _parse_general_metadata(
        self, *, sample_set: str, data: Union[bytes, Exception]
    ) -> pd.DataFrame:
        if isinstance(data, bytes):
            dtype = {
                "sample_id": object,
                "partner_sample_id": object,
                "contributor": object,
                "country": object,
                "location": object,
                "year": "int64",
                "month": "int64",
                "latitude": "float64",
                "longitude": "float64",
                "sex_call": object,
            }
            df = pd.read_csv(io.BytesIO(data), dtype=dtype, na_values="")

            # Ensure all column names are lower case.
            df.columns = [c.lower() for c in df.columns]

            # Add a couple of columns for convenience.
            df["sample_set"] = sample_set
            release = self.lookup_release(sample_set=sample_set)
            df["release"] = release

            # Derive a quarter column from month.
            df["quarter"] = df.apply(
                lambda row: ((row.month - 1) // 3) + 1 if row.month > 0 else -1,
                axis="columns",
            )

            return df

        elif isinstance(data, Exception):
            # Unexpected error.
            raise data

        else:
            raise TypeError

    @doc(
        summary="""
            Read general sample metadata for one or more sample sets into a pandas
            DataFrame.
        """,
        returns="A pandas DataFrame, one row per sample.",
    )
    def general_metadata(
        self, sample_sets: Optional[base_params.sample_sets] = None
    ) -> pd.DataFrame:
        # Normalise input parameters.
        sample_sets_prepped = self._prep_sample_sets_param(sample_sets=sample_sets)
        del sample_sets

        # Obtain paths for all files we need to fetch.
        file_paths: Mapping[str, str] = self._general_metadata_paths(
            sample_sets=sample_sets_prepped
        )

        # Fetch all files. N.B., here is an optimisation, this allows us to fetch
        # multiple files concurrently.
        files: Mapping[str, bytes] = self.read_files(
            paths=file_paths.values(), on_error="return"
        )

        # Parse files into dataframes.
        dfs = []
        for sample_set in sample_sets_prepped:
            path = file_paths[sample_set]
            data = files[path]
            df = self._parse_general_metadata(sample_set=sample_set, data=data)
            dfs.append(df)

        # Concatenate all dataframes.
        df_ret = pd.concat(dfs, axis=0, ignore_index=True)

        return df_ret

    @property
    def _cohorts_analysis(self):
        if self._cohorts_analysis_override:
            return self._cohorts_analysis_override
        else:
            # N.B., this will return None if the key is not present in the
            # config.
            return self.config.get("DEFAULT_COHORTS_ANALYSIS")

    def _cohorts_metadata_paths(self, *, sample_sets: List[str]) -> Dict[str, str]:
        cohorts_analysis = self._cohorts_analysis
        # Guard to ensure this function is only ever called if a cohort
        # analysis is configured for this data resource.
        assert cohorts_analysis
        paths = dict()
        for sample_set in sample_sets:
            release = self.lookup_release(sample_set=sample_set)
            release_path = self._release_to_path(release=release)
            path = f"{release_path}/metadata/cohorts_{cohorts_analysis}/{sample_set}/samples.cohorts.csv"
            paths[sample_set] = path
        return paths

    @property
    def _cohorts_metadata_columns(self):
        # Handle changes to columns used in different analyses.
        cols = None
        if self._cohorts_analysis:
            if self._cohorts_analysis < "20230223":
                cols = (
                    "country_iso",
                    "admin1_name",
                    "admin1_iso",
                    "admin2_name",
                    "taxon",
                    "cohort_admin1_year",
                    "cohort_admin1_month",
                    "cohort_admin2_year",
                    "cohort_admin2_month",
                )
            # We assume that cohorts analyses from "20230223" onwards always include quarter
            # columns.
            else:
                cols = (
                    "country_iso",
                    "admin1_name",
                    "admin1_iso",
                    "admin2_name",
                    "taxon",
                    "cohort_admin1_year",
                    "cohort_admin1_month",
                    "cohort_admin1_quarter",
                    "cohort_admin2_year",
                    "cohort_admin2_month",
                    "cohort_admin2_quarter",
                )
        return cols

    @property
    def _cohorts_metadata_dtype(self):
        cols = self._cohorts_metadata_columns
        if cols:
            # All columns are string columns.
            dtype = {c: object for c in cols}
            dtype["sample_id"] = object
            return dtype

    def _parse_cohorts_metadata(
        self, *, sample_set: str, data: Union[bytes, Exception]
    ) -> pd.DataFrame:
        if isinstance(data, bytes):
            # Parse CSV data.
            dtype = self._cohorts_metadata_dtype
            df = pd.read_csv(io.BytesIO(data), dtype=dtype, na_values="")

            # Ensure all column names are lower case.
            df.columns = [c.lower() for c in df.columns]

            # Rename some columns for consistent naming.
            df.rename(
                columns={
                    "adm1_iso": "admin1_iso",
                    "adm1_name": "admin1_name",
                    "adm2_name": "admin2_name",
                },
                inplace=True,
            )

            return df

        elif isinstance(data, FileNotFoundError):
            # Cohorts metadata are missing for this sample set, fill with a blank
            # DataFrame.

            # Get sample ids as an index via general metadata (TODO caching?).
            df_general = self.general_metadata(sample_sets=sample_set)
            df_general.set_index("sample_id", inplace=True)

            # Create a blank DataFrame with cohorts metadata columns and sample_id index.
            df = pd.DataFrame(
                columns=self._cohorts_metadata_columns,
                index=df_general.index.copy(),
            )

            # Revert sample_id index to column.
            df.reset_index(inplace=True)

            return df

        elif isinstance(data, Exception):
            # Unexpected error.
            raise data

        else:
            raise TypeError

    @doc(
        summary="TODO",
        returns="TODO",
    )
    def cohorts_metadata(
        self, sample_sets: Optional[base_params.sample_sets] = None
    ) -> pd.DataFrame:
        # Not all data resources have cohorts metadata.
        if not self._cohorts_analysis:
            raise NotImplementedError(
                "Cohorts metadata not available for this data resource."
            )

        # Normalise input parameters.
        sample_sets_prepped = self._prep_sample_sets_param(sample_sets=sample_sets)
        del sample_sets

        # Obtain paths for all files we need to fetch.
        file_paths: Mapping[str, str] = self._cohorts_metadata_paths(
            sample_sets=sample_sets_prepped
        )

        # Fetch all files. N.B., here is an optimisation, this allows us to fetch
        # multiple files concurrently.
        files: Mapping[str, bytes] = self.read_files(
            paths=file_paths.values(), on_error="return"
        )

        # Parse files into dataframes.
        dfs = []
        for sample_set in sample_sets_prepped:
            path = file_paths[sample_set]
            data = files[path]
            df = self._parse_cohorts_metadata(sample_set=sample_set, data=data)
            dfs.append(df)

        # Concatenate all dataframes.
        df_ret = pd.concat(dfs, axis=0, ignore_index=True)

        return df_ret

    @property
    def _aim_analysis(self):
        if self._aim_analysis_override:
            return self._aim_analysis_override
        else:
            # N.B., this will return None if the key is not present in the
            # config.
            return self.config.get("DEFAULT_AIM_ANALYSIS")

    def _aim_metadata_paths(self, *, sample_sets: List[str]) -> Dict[str, str]:
        aim_analysis = self._aim_analysis
        # Guard to ensure this function is only ever called if an AIM
        # analysis is configured for this data resource.
        assert aim_analysis
        paths = dict()
        for sample_set in sample_sets:
            release = self.lookup_release(sample_set=sample_set)
            release_path = self._release_to_path(release=release)
            path = f"{release_path}/metadata/species_calls_aim_{aim_analysis}/{sample_set}/samples.species_aim.csv"
            paths[sample_set] = path
        return paths

    def _parse_aim_metadata(
        self, *, sample_set: str, data: Union[bytes, Exception]
    ) -> pd.DataFrame:
        if isinstance(data, bytes):
            # Parse CSV data.
            df = pd.read_csv(
                io.BytesIO(data), dtype=self._aim_metadata_dtype, na_values=""
            )

            # Ensure all column names are lower case.
            df.columns = [c.lower() for c in df.columns]

            return df

        elif isinstance(data, FileNotFoundError):
            # AIM data are missing for this sample set, fill with a blank DataFrame.

            # Get sample ids as an index via general metadata (TODO caching?).
            df_general = self.general_metadata(sample_sets=sample_set)
            df_general.set_index("sample_id", inplace=True)

            # Create a blank DataFrame with AIM metadata columns and sample_id index.
            df = pd.DataFrame(
                columns=self._aim_metadata_columns,
                index=df_general.index.copy(),
            )

            # Revert sample_id index to column.
            df.reset_index(inplace=True)

            return df

        elif isinstance(data, Exception):
            # Unexpected error.
            raise data

        else:
            raise TypeError

    @doc(
        summary="TODO",
        returns="TODO",
    )
    def aim_metadata(
        self, sample_sets: Optional[base_params.sample_sets] = None
    ) -> pd.DataFrame:
        # Not all data resources have AIM data.
        if not self._aim_analysis:
            raise NotImplementedError(
                "AIM metadata not available for this data resource."
            )

        # Normalise input parameters.
        sample_sets_prepped = self._prep_sample_sets_param(sample_sets=sample_sets)
        del sample_sets

        # Obtain paths for all files we need to fetch.
        file_paths: Mapping[str, str] = self._aim_metadata_paths(
            sample_sets=sample_sets_prepped
        )

        # Fetch all files. N.B., here is an optimisation, this allows us to fetch
        # multiple files concurrently.
        files: Mapping[str, bytes] = self.read_files(
            paths=file_paths.values(), on_error="return"
        )

        # Parse files into dataframes.
        dfs = []
        for sample_set in sample_sets_prepped:
            path = file_paths[sample_set]
            data = files[path]
            df = self._parse_aim_metadata(sample_set=sample_set, data=data)
            dfs.append(df)

        # Concatenate all dataframes.
        df_ret = pd.concat(dfs, axis=0, ignore_index=True)

        return df_ret

    @doc(
        summary="TODO",
        returns="TODO",
    )
    def sample_metadata(
        self,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
    ) -> pd.DataFrame:
        # TODO Concurrent reading?
        # TODO Caching?

        # Build a dataframe from all available metadata.
        df_samples = self.general_metadata(sample_sets=sample_sets)
        if self._aim_analysis:
            df_aim = self.aim_metadata(sample_sets=sample_sets)
            df_samples = df_samples.merge(df_aim, on="sample_id", sort=False)
        if self._cohorts_analysis:
            df_cohorts = self.cohorts_metadata(sample_sets=sample_sets)
            df_samples = df_samples.merge(df_cohorts, on="sample_id", sort=False)

        # TODO Extra metadata.

        # For convenience, apply a query.
        if sample_query is not None:
            if isinstance(sample_query, str):
                # Assume a pandas query string.
                df_samples = df_samples.query(sample_query)
            else:
                # Assume it is an indexer.
                df_samples = df_samples.iloc[sample_query]
            df_samples = df_samples.reset_index(drop=True)

        return df_samples.copy()
