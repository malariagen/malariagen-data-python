import io
from collections.abc import Mapping
from typing import Dict, List, Optional

import pandas as pd
from numpydoc_decorator import doc

from .base import AnophelesBase, base_params


class AnophelesSampleData(AnophelesBase):
    def __init__(self, **kwargs):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

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
        files: Mapping[str, bytes] = self.read_files(paths=file_paths.values())

        # Parse files into dataframes.
        dfs = []
        for sample_set in sample_sets_prepped:
            # Parse file data.
            path = file_paths[sample_set]
            data = files[path]
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

            dfs.append(df)

        # Concatenate all dataframes.
        df_ret = pd.concat(dfs, axis=0, ignore_index=True)

        return df_ret
