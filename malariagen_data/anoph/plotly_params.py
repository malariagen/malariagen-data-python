"""Parameters for any plotting functions using plotly."""

# N.B., most of these parameters are always able to take None
# and so we set as Optional here, rather than having to repeat
# that for each function doc.

from typing import List, Literal, Optional, Union

import plotly.graph_objects as go
from typing_extensions import Annotated, TypeAlias

x_label: TypeAlias = Annotated[
    Optional[str],
    "X axis label.",
]

y_label: TypeAlias = Annotated[
    Optional[str],
    "Y axis label.",
]

width: TypeAlias = Annotated[
    Optional[int],
    "Plot width in pixels (px).",
]

height: TypeAlias = Annotated[
    Optional[int],
    "Plot height in pixels (px).",
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

text_auto: TypeAlias = Annotated[
    Union[bool, str],
    """
    If True or a string, single-channel img values will be displayed as text. A
    string like '.2f' will be interpreted as a texttemplate numeric formatting
    directive.
    """,
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
    Optional[str],
    "Name of variable to use to color the markers.",
]

symbol: TypeAlias = Annotated[
    Optional[str],
    "Name of the variable to use to choose marker symbols.",
]

jitter_frac: TypeAlias = Annotated[
    Optional[float],
    "Randomly jitter points by this fraction of their range.",
]

marker_size: TypeAlias = Annotated[
    int,
    "Marker size.",
]

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

figure: TypeAlias = Annotated[
    Optional[go.Figure], "A plotly figure (only returned if show=False)."
]
