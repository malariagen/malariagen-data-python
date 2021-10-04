import json
import os

import fsspec
import pandas as pd
import zarr
from fsspec.mapping import FSMap

from malariagen_data.util import SafeStore, from_zarr


class Pf7:
    """Provides access to data from the Pf7 release"""

    def __init__(self, url, data_config=None, **kwargs):
        self.kwargs = self._set_cloud_access(url, kwargs)
        self._fs, self._path = self._process_url_with_fsspec(url)
        if not data_config:
            working_dir = os.path.dirname(os.path.abspath(__file__))
            data_config = os.path.join(working_dir, "pf7_config.json")
        self.CONF = self._load_data_structure(data_config)
        # setup caches
        self._cache_general_metadata = None
        self._cache_snp_sites = None

    def _set_cloud_access(self, url, kwargs):
        #  special case Google Cloud Storage, use anonymous access, avoids a delay
        if url.startswith("gs://") or url.startswith("gcs://"):
            kwargs["token"] = "anon"
        elif "gs://" in url:
            # chained URL
            kwargs["gs"] = dict(token="anon")
        elif "gcs://" in url:
            # chained URL
            kwargs["gcs"] = dict(token="anon")
        return kwargs

    def _process_url_with_fsspec(self, url):
        fs, path = fsspec.core.url_to_fs(url, **self.kwargs)
        path = path.rstrip("/")
        return fs, path

    def _load_data_structure(self, data_config):
        with open(data_config) as pf7_json_conf:
            config = json.load(pf7_json_conf)
        return config

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
            return self._cache_general_metadata
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

    def open_snp_sites(self):
        if self._cache_snp_sites is None:
            path = os.path.join(self._path, self.CONF["zarr_path"])
            store = SafeStore(FSMap(root=path, fs=self._fs, check=False, create=False))
            """WARNING: Metadata has not been consolidated yet. Using open for now but will eventually switch to opn_consolidated when the .zmetadata file has been created
            """
            root = zarr.open(store=store)
            self._cache_snp_sites = root
        return self._cache_snp_sites

    def snp_sites(
        self,
        field=None,
        site_mask=None,
        site_filters="dt_20200416",
        inline_array=True,
        chunks="native",
    ):

        if field is None:
            # return POS, REF, ALT
            ret = tuple(
                self.snp_sites(field=f, site_mask=None) for f in ("POS", "REF", "ALT")
            )

        else:
            root = self.open_snp_sites()
            z = root["variants"][field]
            ret = from_zarr(z, inline_array=inline_array, chunks=chunks)

        return ret
