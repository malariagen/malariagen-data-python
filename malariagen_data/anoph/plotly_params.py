"""Parameters for any plotting functions using plotly."""

# N.B., most of these parameters are always able to take None
# and so we set as Optional here, rather than having to repeat
# that for each function doc.

from typing import List, Literal, Mapping, Optional, Union

import plotly.graph_objects as go  # type: ignore
from typing_extensions import Annotated, TypeAlias

x_label: TypeAlias = Annotated[
    Optional[str],
    "X axis label.",
]

y_label: TypeAlias = Annotated[
    Optional[str],
    "Y axis label.",
]

fig_width: TypeAlias = Annotated[
    Optional[int],
    "Figure width in pixels (px).",
]

fig_height: TypeAlias = Annotated[
    Optional[int],
    "Figure weight in pixels (px).",
]

height: TypeAlias = Annotated[
    int,
    "Height in pixels (px).",
]

aspect: TypeAlias = Annotated[
    Optional[Literal["equal", "auto"]],
    "Aspect ratio, see also https://plotly.com/python-api-reference/generated/plotly.express.imshow",
]

title: TypeAlias = Annotated[
    Optional[Union[str, bool]],
    """
    If True, attempt to use metadata from input dataset as a plot title.
    Otherwise, use supplied value as a title.
    """,
]

title_font_size = Annotated[int, "Font size for the plot title."]

text_auto: TypeAlias = Annotated[
    Union[bool, str],
    """
    If True or a string, single-channel img values will be displayed as text. A
    string like '.2f' will be interpreted as a texttemplate numeric formatting
    directive.
    """,
]

color_discrete_sequence: TypeAlias = Annotated[
    Optional[List], "Provide a list of colours to use."
]

color_discrete_map: TypeAlias = Annotated[
    Optional[Mapping], "Provide an explicit mapping from values to colours."
]

category_order: TypeAlias = Annotated[
    Optional[Union[List, Mapping]],
    "Control the order in which values appear in the legend.",
]

color_continuous_scale: TypeAlias = Annotated[
    Optional[Union[str, List[str]]],
    """
    Colormap used to map scalar data to colors (for a 2D image). This
    parameter is not used for RGB or RGBA images. If a string is provided,
    it should be the name of a known color scale, and if a list is provided,
    it should be a list of CSS-compatible colors.
    """,
]

colorbar: TypeAlias = Annotated[
    bool,
    "If False, do not display a color bar.",
]

x: TypeAlias = Annotated[
    str,
    "Name of variable to plot on the X axis.",
]

y: TypeAlias = Annotated[
    str,
    "Name of variable to plot on the Y axis.",
]

z: TypeAlias = Annotated[
    str,
    "Name of variable to plot on the Z axis.",
]

color: TypeAlias = Annotated[
    Optional[Union[str, Mapping]],
    "Name of variable to use to color the markers.",
]

symbol: TypeAlias = Annotated[
    Optional[Union[str, Mapping]],
    "Name of the variable to use to choose marker symbols.",
]

jitter_frac: TypeAlias = Annotated[
    Optional[float],
    "Randomly jitter points by this fraction of their range.",
]

marker_size: TypeAlias = Annotated[
    Union[int, float],
    "Marker size.",
]

line_width: TypeAlias = Annotated[Union[int, float], "Line width."]

line_color: TypeAlias = Annotated[str, "Line color"]

template: TypeAlias = Annotated[
    Optional[
        Literal[
            "ggplot2",
            "seaborn",
            "simple_white",
            "plotly",
            "plotly_white",
            "plotly_dark",
            "presentation",
            "xgridoff",
            "ygridoff",
            "gridon",
            "none",
        ]
    ],
    "The figure template name (must be a key in plotly.io.templates).",
]

show: TypeAlias = Annotated[
    bool,
    "If true, show the plot. If False, do not show the plot, but return the figure.",
]

renderer: TypeAlias = Annotated[Optional[str], "The name of the renderer to use."]

render_mode: TypeAlias = Annotated[
    Literal["auto", "svg", "webgl"],
    "The type of rendering backend to use. See also https://plotly.com/python/webgl-vs-svg/",
]

render_mode_default: render_mode = "auto"

figure: TypeAlias = Annotated[
    Optional[go.Figure], "A plotly figure (only returned if show=False)."
]

zmin: TypeAlias = Annotated[
    Union[int, float],
    "The lower end of the range of values that the colormap covers.",
]

zmax: TypeAlias = Annotated[
    Union[int, float],
    "The upper end of the range of values that the colormap covers.",
]

legend_sizing: TypeAlias = Annotated[
    Literal["constant", "trace"],
    """
    Determines if the legend items symbols scale with their corresponding
    "trace" attributes or remain "constant" independent of the symbol size
    on the graph.
    """,
]
