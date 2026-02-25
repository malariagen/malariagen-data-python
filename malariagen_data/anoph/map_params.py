"""Parameters for functions plotting maps using ipyleaflet."""

import warnings
from typing import Dict, Tuple, Union

import ipyleaflet  # type: ignore
import xyzservices  # type: ignore
from typing_extensions import Annotated, TypeAlias

center: TypeAlias = Annotated[
    Tuple[Union[int, float], Union[int, float]],
    "Location to center the map.",
]

center_default: center = (-2, 20)

zoom: TypeAlias = Annotated[Union[int, float], "Initial zoom level."]

zoom_default: zoom = 3

# Candidate basemap abbreviations — loaded lazily to avoid import failures
# if a provider has been decommissioned upstream.
_basemap_abbrev_candidates = {
    "mapnik": lambda: ipyleaflet.basemaps.OpenStreetMap.Mapnik,
    "natgeoworldmap": lambda: ipyleaflet.basemaps.Esri.NatGeoWorldMap,
    "opentopomap": lambda: ipyleaflet.basemaps.OpenTopoMap,
    "positron": lambda: ipyleaflet.basemaps.CartoDB.Positron,
    "satellite": lambda: ipyleaflet.basemaps.Gaode.Satellite,
    "worldimagery": lambda: ipyleaflet.basemaps.Esri.WorldImagery,
    "worldstreetmap": lambda: ipyleaflet.basemaps.Esri.WorldStreetMap,
    "worldtopomap": lambda: ipyleaflet.basemaps.Esri.WorldTopoMap,
}

_basemap_abbrevs: dict | None = None


def get_basemap_abbrevs() -> dict:
    """Return available basemap abbreviations, skipping any unavailable providers."""
    global _basemap_abbrevs
    if _basemap_abbrevs is None:
        _basemap_abbrevs = {}
        for key, provider_fn in _basemap_abbrev_candidates.items():
            try:
                _basemap_abbrevs[key] = provider_fn()
            except Exception:
                warnings.warn(
                    f"Basemap provider {key!r} is not available and will be skipped.",
                    stacklevel=2,
                )
    return _basemap_abbrevs


# Keep basemap_abbrevs as a property-like alias for backwards compatibility.
# Code that does `map_params.basemap_abbrevs` will call get_basemap_abbrevs().
class _BasemapAbbrevProxy:
    def __getattr__(self, item):
        return getattr(get_basemap_abbrevs(), item)

    def __iter__(self):
        return iter(get_basemap_abbrevs())

    def __getitem__(self, item):
        return get_basemap_abbrevs()[item]

    def __contains__(self, item):
        return item in get_basemap_abbrevs()

    def keys(self):
        return get_basemap_abbrevs().keys()

    def values(self):
        return get_basemap_abbrevs().values()

    def items(self):
        return get_basemap_abbrevs().items()


basemap_abbrevs = _BasemapAbbrevProxy()

basemap: TypeAlias = Annotated[
    Union[str, Dict, ipyleaflet.TileLayer, xyzservices.lib.TileProvider],
    f"""
    Basemap from ipyleaflet or other TileLayer provider. Strings are abbreviations mapped to corresponding
    basemaps, available values are {list(_basemap_abbrev_candidates.keys())}.
    """,
]

basemap_default: basemap = "mapnik"

height: TypeAlias = Annotated[
    Union[int, str], "Height of the map in pixels (px) or other units."
]

height_default: height = 500

width: TypeAlias = Annotated[
    Union[int, str], "Width of the map in pixels (px) or other units."
]

width_default: width = "100%"
