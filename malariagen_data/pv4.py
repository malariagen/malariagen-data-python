import os

from malariagen_data.plasmodium import PlasmodiumTools


class Pv4:
    """Provides access to data from the Pv4 release.
    Parameters
    ----------
    url : str, optional
        Base path to data. Default uses Google Cloud Storage "gs://pv4_release/",
        or specify a local path on your file system if data have been downloaded.
    data_config : str, optional
        Path to config for structure of Pv4 data resource. Defaults to config included
        with the malariagen_data package.
    **kwargs
        Passed through to fsspec when setting up file system access.

    Examples
    --------
    Access data from Google Cloud Storage (default):

        >>> import malariagen_data
        >>> pv4 = malariagen_data.Pv4()

    Access data downloaded to a local file system:

        >>> pv4 = malariagen_data.Pv4("/local/path/to/pv4_release/")

    """

    def __init__(
        self,
        url=None,
        data_config=None,
        **kwargs,
    ):

        if not data_config:
            working_dir = os.path.dirname(os.path.abspath(__file__))
            data_config = os.path.join(working_dir, "pv4_config.json")
        self.tools = PlasmodiumTools(data_config, url)

    def sample_metadata(self):
        """Access sample metadata and return as pandas dataframe.

        Returns
        -------
        df : pandas.DataFrame
            A dataframe of sample metadata on the samples that were sequenced as part of this resource.
            Includes the time and place of collection, quality metrics, and accesion numbers.
            One row per sample.

        Example
        -------
        Access metadata as pandas dataframe:

            >>> pv4.sample_metadata()

        """

        return self.tools.open_sample_metadata()

    def variant_calls(self, extended=False, inline_array=True, chunks="native"):
        """Access variant sites, site filters and genotype calls.

        Parameters
        ----------
        extended : bool, optional
            If False only the default variables are returned. If True all variables from the zarr are returned. Defaults to False.
        inline_array : bool, optional
            Passed through to dask.array.from_array(). Defaults to True.
        chunks : str, optional
            If 'auto' let dask decide chunk size. If 'native' use native zarr chunks.
            Also can be a target size, e.g., '200 MiB'. Defaults to "native".

        Returns
        -------
        ds : xarray.Dataset
            Dataset containing either default or extended variables from the variant calls Zarr.

        Examples
        --------
        Access core set of variables for variant calls (default):

            >>> pv4.variant_calls()

        Access extended set of variables for variant calls:

            >>> pv4.variant_calls(extended=True)

        """
        ds = self.tools.load_variant_calls(
            extended=extended, inline_array=inline_array, chunks=chunks
        )
        return ds
