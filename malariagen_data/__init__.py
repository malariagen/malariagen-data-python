# flake8: noqa
from .ag3 import Ag3, Region
from .amin1 import Amin1
from .util import SiteClass

try:
    import importlib.metadata as importlib_metadata
except ModuleNotFoundError:
    import importlib_metadata


# this will read version from pyproject.toml
__version__ = importlib_metadata.version(__name__)
