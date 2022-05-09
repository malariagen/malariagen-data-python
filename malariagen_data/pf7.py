import os

from .plasmodium import PlasmodiumDataResource


class Pf7(PlasmodiumDataResource):
    """Provides access to data from the Pf7 release.

    Parameters
    ----------
    url : str, optional
        Base path to data. Default uses Google Cloud Storage "gs://pf7_release/",
        or specify a local path on your file system if data have been downloaded.
    data_config : str, optional
        Path to config for structure of Pf7 data resource. Defaults to config included
        with the malariagen_data package.
    **kwargs
        Passed through to fsspec when setting up file system access.

    Examples
    --------
    Access data from Google Cloud Storage (default):

        >>> import malariagen_data
        >>> pf7 = malariagen_data.Pf7()

    Access data downloaded to a local file system:

        >>> pf7 = malariagen_data.Pf7("/local/path/to/pf7_release/")

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
            data_config = os.path.join(working_dir, "pf7_config.json")
        super().__init__(data_config=data_config, url=url)
