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

basemap_abbrevs = {
    "mapnik": ipyleaflet.basemaps.OpenStreetMap.Mapnik,
    "natgeoworldmap": ipyleaflet.basemaps.Esri.NatGeoWorldMap,
    "opentopomap": ipyleaflet.basemaps.OpenTopoMap,
    "positron": ipyleaflet.basemaps.CartoDB.Positron,
    "satellite": ipyleaflet.basemaps.Gaode.Satellite,
    "worldimagery": ipyleaflet.basemaps.Esri.WorldImagery,
    "worldstreetmap": ipyleaflet.basemaps.Esri.WorldStreetMap,
    "worldtopomap": ipyleaflet.basemaps.Esri.WorldTopoMap,
}

basemap: TypeAlias = Annotated[
    Union[str, Dict, ipyleaflet.TileLayer, xyzservices.lib.TileProvider],
    f"""
    Basemap from ipyleaflet or other TileLayer provider. Strings are abbreviations mapped to corresponding
    basemaps, available values are {list(basemap_abbrevs.keys())}.
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
