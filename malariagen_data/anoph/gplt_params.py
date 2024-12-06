"""Parameters for genome plotting functions. N.B., genome plots are always
plotted with bokeh."""

from typing import Literal, Mapping, Optional, Union, Final, Sequence

import bokeh.models
from typing_extensions import Annotated, TypeAlias

sizing_mode: TypeAlias = Annotated[
    Literal[
        "fixed",
        "stretch_width",
        "stretch_height",
        "stretch_both",
        "scale_width",
        "scale_height",
        "scale_both",
    ],
    """
    Bokeh plot sizing mode, see also
    https://docs.bokeh.org/en/latest/docs/user_guide/basic/layouts.html#sizing-modes
    """,
]

sizing_mode_default: sizing_mode = "stretch_width"

width: TypeAlias = Annotated[
    Optional[int],  # always can be None
    "Plot width in pixels (px).",
]

width_default: width = None

height: TypeAlias = Annotated[
    int,
    "Plot height in pixels (px).",
]

row_height: TypeAlias = Annotated[
    int,
    "Plot height per row (sample) in pixels (px).",
]

track_height: TypeAlias = Annotated[
    int,
    "Main track height in pixels (px).",
]

genes_height: TypeAlias = Annotated[
    int,
    "Genes track height in pixels (px).",
]

genes_height_default: genes_height = 90

show: TypeAlias = Annotated[
    bool,
    "If true, show the plot. If False, do not show the plot, but return the figure.",
]

toolbar_location: TypeAlias = Annotated[
    Literal["above", "below", "left", "right"],
    "Location of bokeh toolbar.",
]

toolbar_location_default: toolbar_location = "above"

x_range: TypeAlias = Annotated[
    bokeh.models.Range,
    "X axis range (for linking to other tracks).",
]

title: TypeAlias = Annotated[
    Union[str, bool],
    "Plot title. If True, a title may be automatically generated.",
]

figure: TypeAlias = Annotated[
    # Use quite a broad type here to accommodate both single-panel figures
    # created via bokeh.plotting and multi-panel figures created via
    # bokeh.layouts.
    bokeh.model.Model,
    "A bokeh figure.",
]

optional_figure: TypeAlias = Annotated[
    # Use quite a broad type here to accommodate both single-panel figures
    # created via bokeh.plotting and multi-panel figures created via
    # bokeh.layouts.
    Optional[figure],
    "A bokeh figure (only returned if show=False).",
]

output_backend: TypeAlias = Annotated[
    Literal["canvas", "webgl", "svg"],
    """
    Specify an output backend to render a plot area onto. See also
    https://docs.bokeh.org/en/latest/docs/user_guide/output/webgl.html
    """,
]

# webgl is better for plots like selection scans with lots of points
output_backend_default: output_backend = "webgl"

circle_kwargs: TypeAlias = Annotated[
    Mapping,
    "Passed through to bokeh scatter() function with marker = 'circle'.",
]

line_kwargs: TypeAlias = Annotated[
    Mapping,
    "Passed through to bokeh line() function.",
]

contig_colors: TypeAlias = Annotated[
    list[str],
    "A sequence of colors.",
]

contig_colors_default: Final[contig_colors] = list(bokeh.palettes.d3["Category20b"][5])

colors: TypeAlias = Annotated[Sequence[str], "List of colors."]

gene_labels: TypeAlias = Annotated[
    Mapping[str, str],
    "A mapping of gene identifiers to custom labels, which will appear in the plot.",
]

gene_labelset: TypeAlias = Annotated[
    bokeh.models.LabelSet,  # type: ignore
    "A LabelSet to use in the plot.",
]
