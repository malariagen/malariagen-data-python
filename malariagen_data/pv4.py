import os

from malariagen_data.plasmodium import PlasmodiumDataResource


class Pv4(PlasmodiumDataResource):
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

        # setup filesystem
        if not data_config:
            working_dir = os.path.dirname(os.path.abspath(__file__))
            data_config = os.path.join(working_dir, "pv4_config.json")
        super().__init__(data_config=data_config, url=url)
