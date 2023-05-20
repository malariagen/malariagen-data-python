"""Parameters for functions plotting maps using ipyleaflet."""

from typing import Dict, Tuple, Union

import ipyleaflet
import xyzservices
from typing_extensions import Annotated, TypeAlias

center: TypeAlias = Annotated[
    Tuple[int, int],
    "Location to center the map.",
]

center_default: center = (-2, 20)

zoom: TypeAlias = Annotated[int, "Initial zoom level."]

zoom_default: zoom = 3

basemap: TypeAlias = Annotated[
    Union[str, Dict, ipyleaflet.TileLayer, xyzservices.lib.TileProvider],
    """
    Basemap from ipyleaflet or other TileLayer provider. Strings are abbreviations mapped to corresponding
    basemaps, e.g. "mapnik" (case-insensitive) maps to TileProvider ipyleaflet.basemaps.OpenStreetMap.Mapnik.
    """,
]

basemap_default: basemap = "mapnik"

basemap_abbrevs = {
    "mapnik": ipyleaflet.basemaps.OpenStreetMap.Mapnik,
    "natgeoworldmap": ipyleaflet.basemaps.Esri.NatGeoWorldMap,
    "opentopomap": ipyleaflet.basemaps.OpenTopoMap,
    "positron": ipyleaflet.basemaps.CartoDB.Positron,
    "satellite": ipyleaflet.basemaps.Gaode.Satellite,
    "terrain": ipyleaflet.basemaps.Stamen.Terrain,
    "watercolor": ipyleaflet.basemaps.Stamen.Watercolor,
    "worldimagery": ipyleaflet.basemaps.Esri.WorldImagery,
    "worldstreetmap": ipyleaflet.basemaps.Esri.WorldStreetMap,
    "worldtopomap": ipyleaflet.basemaps.Esri.WorldTopoMap,
}

height: TypeAlias = Annotated[
    Union[int, str], "Height of the map in pixels (px) or other units."
]

height_default: height = 500

width: TypeAlias = Annotated[
    Union[int, str], "Width of the map in pixels (px) or other units."
]

width_default: width = "100%"
