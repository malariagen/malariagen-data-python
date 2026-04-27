import os

from .plasmodium import PlasmodiumDataResource


class Pf9(PlasmodiumDataResource):
    """Provides access to data from the Pf9 release.

    Parameters
    ----------
    url : str, optional
        Base path to data. Default uses Google Cloud Storage "gs://pf9-release/",
        or specify a local path on your file system if data have been downloaded.
    data_config : str, optional
        Path to config for structure of Pf9 data resource. Defaults to config included
        with the malariagen_data package.
    **kwargs
        Passed through to fsspec when setting up file system access.

    Examples
    --------
    Access data from Google Cloud Storage (default):

        >>> import malariagen_data
        >>> pf9 = malariagen_data.Pf9()

    Access data downloaded to a local file system:

        >>> pf9 = malariagen_data.Pf9("/local/path/to/pf9-release/")

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
            data_config = os.path.join(working_dir, "pf9_config.json")
        super().__init__(data_config=data_config, url=url)
