"""Parameters for functions using plotly dash (e.g., haplotype networks)."""

from typing import Literal, Union

from typing_extensions import Annotated, TypeAlias

height: TypeAlias = Annotated[int, "Height of the Dash app in pixels (px)."]

width: TypeAlias = Annotated[Union[int, str], "Width of the Dash app."]

server_mode: TypeAlias = Annotated[
    Literal["inline", "external", "jupyterlab", "tab"],
    """
    Controls how the Jupyter Dash app will be launched. See
    https://dash.plotly.com/dash-in-jupyter#display-modes for
    more information.
    """,
]

server_mode_default: server_mode = "inline"

server_port: TypeAlias = Annotated[
    int,
    "Manually override the port on which the Dash app will run.",
]

serve_scripts_locally: TypeAlias = Annotated[
    bool, "If True, serve scripts locally, otherwise use CDN."
]

serve_scripts_locally_default: serve_scripts_locally = True
