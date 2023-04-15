import io
from typing import Dict, List

import pandas as pd

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

    def general_metadata(self, sample_sets: base_params.sample_sets = None):
        sample_sets_prepped = self._prep_sample_sets_param(sample_sets=sample_sets)
        paths = self._general_metadata_paths(sample_sets=sample_sets_prepped)
        files = self.read_files(paths=list(paths.values()))
        dfs = []
        for sample_set in sample_sets:
            path = paths[sample_set]
            data = files[path]
            df = pd.read_csv(io.BytesIO(data))
            dfs.append(df)
        df_ret = pd.concat(dfs, axis=0, ignore_index=True)
        return df_ret
