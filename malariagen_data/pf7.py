import json
import os

import fsspec
import pandas as pd
import zarr
from fsspec.mapping import FSMap

from .util import SafeStore, from_zarr


class Pf7:
    """Provides access to data from Pf7 releases.

    Parameters
    ----------
    url : str
        Base path to data. Give "gs://pf7_release/" to use Google Cloud Storage,
        or a local path on your file system if data have been downloaded.
    **kwargs
        Passed through to fsspec when setting up file system access.

    Examples
    --------
    Access data from Google Cloud Storage:

        >>> import malariagen_data
        >>> pf7 = malariagen_data.Pf7("gs://pf7_release/")

    Access data downloaded to a local file system:

        >>> pf7 = malariagen_data.Pf7("/local/path/to/pf7_release/")
    """

    def __init__(self, url, data_config=None, **kwargs):
        self.kwargs = self._set_cloud_access(url, kwargs)
        self._fs, self._path = self._process_url_with_fsspec(url)
        if not data_config:
            working_dir = os.path.dirname(os.path.abspath(__file__))
            data_config = os.path.join(working_dir, "pf7_config.json")
        self.CONF = self._load_data_structure(data_config)

        # setup caches
        self._cache_general_metadata = None
        self._cache_zarr = None

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

    def open_zarr(self):
        """Open SNP sites zarr.

        Returns
        -------
        root : zarr.hierarchy.Group

        """

        if self._cache_zarr is None:
            path = os.path.join(self._path, self.CONF["zarr_path"])
            store = SafeStore(FSMap(root=path, fs=self._fs, check=False, create=False))
            """WARNING: Metadata has not been consolidated yet. Using open for now but will eventually switch to opn_consolidated when the .zmetadata file has been created
            """
            root = zarr.open(store=store)
            self._cache_zarr = root
        return self._cache_zarr

    def variants(
        self,
        field=None,
        inline_array=True,
        chunks="native",
    ):
        """Access SNP site data (positions and alleles).

        Args:
            field (str, optional): {"CHROM", "POS"}, optional
                Array to access. If not provided, all three arrays POS, REF, ALT will be returned as a
                tuple.
            inline_array (bool, optional): Passed through to dask.array.from_array().
                Defaults to True.
            chunks (str, optional): If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
                Also can be a target size, e.g., '200 MiB'. Defaults to "native".

        Returns:
            d : dask.array.Array or tuple of dask.array.Array
        """

        if field is None:
            # if no field specified return CHROM, POS
            ret = tuple(self.variants(field=f) for f in ("CHROM", "POS"))

        else:
            root = self.open_zarr()
            z = root["variants"][field]
            ret = from_zarr(z, inline_array=inline_array, chunks=chunks)

        return ret

    def calldata(
        self,
        field="GT",
        inline_array=True,
        chunks="native",
    ):

        """Access SNP genotypes and associated data.

        Parameters
        ----------
        field : {"GT", "GQ", "AD", "MQ"}
            Array to access.
        inline_array : bool, optional
            Passed through to dask.array.from_array().
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'.

        Returns
        -------
        d : dask.array.Array

        """

        root = self.open_zarr()
        z = root["calldata"][field]
        d = from_zarr(z, inline_array=inline_array, chunks=chunks)
        return d
