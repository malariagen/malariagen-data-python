import json
import os

import pandas as pd
from fsspec.core import url_to_fs


class Pf7:
    """Provides access to data from the Pf7 release"""

    def __init__(self, url, **kwargs):

        # special case Google Cloud Storage, use anonymous access, avoids a delay
        if url.startswith("gs://") or url.startswith("gcs://"):
            kwargs["token"] = "anon"
        elif "gs://" in url:
            # chained URL
            kwargs["gs"] = dict(token="anon")
        elif "gcs://" in url:
            # chained URL
            kwargs["gcs"] = dict(token="anon")

        # process the url using fsspec
        self._pre = kwargs.pop("pre", False)
        fs, path = url_to_fs(url, **kwargs)
        self._fs = fs
        # path compatibility, fsspec/gcsfs behaviour varies between version
        while path.endswith("/"):
            path = path[:-1]
        self._path = path

        # load config with data structure for pf7
        with open("pf7_config.json") as pf7_json_conf:
            self.CONF = json.load(pf7_json_conf)

        # setup caches
        self._cache_general_metadata = None

    def _read_general_metadata(self):
        """Read metadata file.

        Returns:
            df: pandas.DataFrame
        """
        if self._cache_general_metadata is None:
            path = os.path.join(self._path, self.CONF["metadata_path"])
            with self._fs.open(path) as f:
                df = pd.read_csv(f, sep="\t", na_values="")
            self._cache_general_metadata = df
            return df
        else:
            return self._cache_general_metadata

    def sample_metadata(self):
        # TODO Any filtering could be added to this function eventually
        """Access sample metadata.

        Returns:
            df : pandas.DataFrame
        """

        df = self._read_general_metadata()
        return df
