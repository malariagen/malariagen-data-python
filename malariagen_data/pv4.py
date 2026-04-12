import os
from functools import lru_cache

from .plasmodium import PlasmodiumDataResource


"""
Optimized Pv4 Data Resource Class

What’s improved:
1. Cached default config path:
   - The path to 'pv4_config.json' is computed only once using lru_cache.
   - Avoids repeated calls to os.path.abspath and os.path.dirname
     when multiple Pv4 instances are created.

2. Cleaner fallback logic:
   - Uses Pythonic 'data_config or default' pattern instead of explicit condition.

3. Proper kwargs forwarding:
   - Ensures any additional filesystem arguments (**kwargs) are passed
     to the parent class (important for fsspec configurations like authentication, caching, etc.)

Overall Impact:
- Slight performance improvement (micro-optimization)
- Cleaner and more maintainable code
- Better alignment with the documented behavior
"""


@lru_cache(maxsize=1)
def _get_default_config_path():
    """
    Compute and cache the default path to pv4_config.json.

    This avoids recomputing the file path every time a Pv4 object is created.
    """
    working_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(working_dir, "pv4_config.json")


class Pv4(PlasmodiumDataResource):
    """Provides access to data from the Pv4 release.

    Parameters
    ----------
    url : str, optional
        Base path to data. Default uses Google Cloud Storage "gs://pv4_release/",
        or specify a local path if data have been downloaded.
    data_config : str, optional
        Path to config for structure of Pv4 data resource.
        Defaults to packaged pv4_config.json.
    **kwargs
        Additional arguments passed to filesystem backend (via fsspec).

    Examples
    --------
    >>> import malariagen_data
    >>> pv4 = malariagen_data.Pv4()

    >>> pv4 = malariagen_data.Pv4("/local/path/to/pv4_release/")
    """

    def __init__(self, url=None, data_config=None, **kwargs):
        # Use cached default config path if none is provided
        data_config = data_config or _get_default_config_path()

        # Initialize parent class with full argument support
        super().__init__(data_config=data_config, url=url, **kwargs)
