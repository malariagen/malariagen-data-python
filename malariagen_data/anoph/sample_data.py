import io
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from numpydoc_decorator import doc

from .base import AnophelesBase, base_params


class AnophelesSampleData(AnophelesBase):
    def __init__(
        self,
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
        self._aim_analysis_override = aim_analysis

        # N.B., the expected AIM metadata columns may vary between
        # data resources, and so column names and dtype need to be
        # passed in as parameters.
        self._aim_metadata_columns = aim_metadata_columns
        self._aim_metadata_dtype = aim_metadata_dtype

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
        if isinstance(data, Exception):
            # Unexpected error.
            raise data

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

    def _aim_analysis(self):
        if self._aim_analysis_override:
            return self._aim_analysis_override
        else:
            return self.config.get("DEFAULT_AIM_ANALYSIS")

    def _aim_metadata_paths(self, *, sample_sets: List[str]) -> Dict[str, str]:
        aim_analysis = self._aim_analysis
        # Guard to ensure this function is only ever called if an AIM
        # analysis is configured.
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
        # TODO Do we still need to handle the possibility of missing files?

        if isinstance(data, FileNotFoundError):
            # AIM data are missing, fill with a blank DataFrame.

            # Get sample ids as an index via general metadata (TODO caching?).
            df_general = self.general_metadata(sample_sets=sample_set)
            df_general.set_index("sample_id", inplace=True)

            # Create a blank DataFrame with AIM metadata columns and sample_id index.
            df = pd.DataFrame(
                columns=self._aim_metadata_columns,
                dtype=self._aim_metadata_dtype,
                index=df_general.index.copy(),
            )

            # Revert sample_id index to column.
            df.reset_index(inplace=True)

        elif isinstance(data, Exception):
            # Unexpected error.
            raise data

        else:
            df = pd.read_csv(
                io.BytesIO(data), dtype=self._aim_metadata_dtype, na_values=""
            )

        # Ensure all column names are lower case.
        df.columns = [c.lower() for c in df.columns]

        return df

    @doc(
        summary="TODO",
        returns="TODO",
    )
    def aim_metadata(
        self, sample_sets: Optional[base_params.sample_sets] = None
    ) -> pd.DataFrame:
        # Not all data resources have AIM data.
        if not self.aim_analysis:
            raise NotImplementedError("Data not available.")

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
