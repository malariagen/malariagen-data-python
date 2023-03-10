# flake8: noqa
from .af1 import Af1
from .ag3 import Ag3
from .amin1 import Amin1
from .anopheles import AnophelesDataResource, Region
from .pf7 import Pf7
from .pv4 import Pv4
from .util import SiteClass

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata  # type: ignore

# this will read version from pyproject.toml
__version__ = importlib_metadata.version(__name__)


def _ensure_xarray_zarr_backend():
    # workaround for https://github.com/malariagen/malariagen-data-python/issues/320
    # see also https://github.com/pydata/xarray/issues/7478

    from xarray.backends.plugins import list_engines
    from xarray.backends.zarr import ZarrBackendEntrypoint

    # zarr is a dependency, so will always be available if malariagen_data
    # is installed
    ZarrBackendEntrypoint.available = True

    # ensure xarray refreshes list of available engines
    list_engines.cache_clear()


_ensure_xarray_zarr_backend()
