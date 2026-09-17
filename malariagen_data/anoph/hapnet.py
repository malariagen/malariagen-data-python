from typing import Any, Dict, Mapping, Optional

import allel  # type: ignore
import numpy as np
from numpydoc_decorator import doc  # type: ignore

from ..mjn import _median_joining_network, _mjn_graph
from ..util import _check_types, _plotly_discrete_legend
from .safe_query import validate_query
from . import base_params, dash_params, hapnet_params, plotly_params
from .hap_data import AnophelesHapData, hap_params


class AnophelesHapNetAnalysis(
    AnophelesHapData,
):
    def __init__(
        self,
        **kwargs,
    ):
        # N.B., this class is designed to work cooperatively, and
        # so it's important that any remaining parameters are passed
        # to the superclass constructor.
        super().__init__(**kwargs)

    @_check_types
    @doc(
        summary="""
            Construct a median-joining haplotype network and display it using
            Cytoscape.
        """,
        extended_summary="""
            A haplotype network provides a visualisation of the genetic distance
            between haplotypes. Each node in the network represents a unique
            haplotype. The size (area) of the node is scaled by the number of
            times that unique haplotype was observed within the selected samples.
            A connection between two nodes represents a single SNP difference
            between the corresponding haplotypes.
        """,
    )
    def plot_haplotype_network(
        self,
        region: base_params.regions,
        analysis: hap_params.analysis = base_params.DEFAULT,
        sample_sets: Optional[base_params.sample_sets] = None,
        sample_query: Optional[base_params.sample_query] = None,
        sample_query_options: Optional[base_params.sample_query_options] = None,
        max_dist: hapnet_params.max_dist = hapnet_params.max_dist_default,
        color: plotly_params.color = None,
        color_discrete_sequence: plotly_params.color_discrete_sequence = None,
        color_discrete_map: plotly_params.color_discrete_map = None,
        category_orders: plotly_params.category_order = None,
        node_size_factor: hapnet_params.node_size_factor = hapnet_params.node_size_factor_default,
        layout: hapnet_params.layout = hapnet_params.layout_default,
        layout_params: Optional[hapnet_params.layout_params] = None,
        server_port: Optional[dash_params.server_port] = None,
        server_mode: Optional[
            dash_params.server_mode
        ] = dash_params.server_mode_default,
        height: dash_params.height = 600,
        width: Optional[dash_params.width] = "100%",
        serve_scripts_locally: dash_params.serve_scripts_locally = dash_params.serve_scripts_locally_default,
        chunks: base_params.chunks = base_params.native_chunks,
        inline_array: base_params.inline_array = base_params.inline_array_default,
    ):
        import dash_cytoscape as cyto  # type: ignore
        from dash import Dash, dcc, html  # type: ignore
        from dash.dependencies import Input, Output  # type: ignore

        debug = self._log.debug

        # https://dash.plotly.com/dash-in-jupyter#troubleshooting
        # debug("infer jupyter proxy config")
        # Turn off for now, this seems to crash the kernel!
        # from dash import jupyter_dash
        # jupyter_dash.infer_jupyter_proxy_config()

        if layout != "cose":
            cyto.load_extra_layouts()

        debug("access haplotypes dataset")
        ds_haps = self.haplotypes(
            region=region,
            sample_sets=sample_sets,
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            analysis=analysis,
            chunks=chunks,
            inline_array=inline_array,
        )

        debug("access sample metadata")
        df_samples = self.sample_metadata(
            sample_query=sample_query,
            sample_query_options=sample_query_options,
            sample_sets=sample_sets,
        )

        debug("setup haplotype metadata")
        samples_phased = ds_haps["sample_id"].values
        df_samples_phased = (
            df_samples.set_index("sample_id").loc[samples_phased].reset_index()
        )
        df_haps = df_samples_phased.loc[df_samples_phased.index.repeat(2)].reset_index(
            drop=True
        )

        debug("load haplotypes")
        gt = allel.GenotypeDaskArray(ds_haps["call_genotype"].data)
        with self._dask_progress(desc="Load haplotypes"):
            ht = gt.to_haplotypes().compute()

        with self._spinner(desc="Compute haplotype network"):
            debug("count alleles and select segregating sites")
            ac = ht.count_alleles(max_allele=1)
            loc_seg = ac.is_segregating()
            ht_seg = ht[loc_seg]

            debug("identify distinct haplotypes")
            ht_distinct_sets = ht_seg.distinct()
            # find indices of distinct haplotypes - just need one per set
            ht_distinct_indices = [min(s) for s in ht_distinct_sets]
            # reorder by index - TODO is this necessary?
            ix = np.argsort(ht_distinct_indices)
            ht_distinct_indices = [ht_distinct_indices[i] for i in ix]
            ht_distinct_sets = [ht_distinct_sets[i] for i in ix]
            # obtain an array of distinct haplotypes
            ht_distinct = ht_seg.take(ht_distinct_indices, axis=1)
            # count how many observations per distinct haplotype
            ht_counts = [len(s) for s in ht_distinct_sets]

            debug("construct median joining network")
            ht_distinct_mjn, edges, alt_edges = _median_joining_network(
                ht_distinct, max_dist=max_dist
            )
            edges = np.triu(edges)
            alt_edges = np.triu(alt_edges)

            debug("setup colors")
            color_values = None
            color_values_display = None
            color_discrete_map_display = None
            ht_color_counts = None

            if color is not None:
                # Handle string case (direct column name or cohorts_ prefix)
                if isinstance(color, str):
                    # Try direct column name
                    if color in df_haps.columns:
                        color_column = color
                    # Try with cohorts_ prefix
                    elif f"cohorts_{color}" in df_haps.columns:
                        color_column = f"cohorts_{color}"
                    # Neither exists, raise helpful error
                    else:
                        available_columns = ", ".join(df_haps.columns)
                        raise ValueError(
                            f"Column '{color}' or 'cohorts_{color}' not found in sample data. "
                            f"Available columns: {available_columns}"
                        )

                    # Now use the validated color_column for processing
                    df_haps["_partition"] = (
                        df_haps[color_column]
                        .astype(str)
                        .str.replace(r"\W", "", regex=True)
                    )

                    # extract all unique values of the color column
                    color_values = df_haps["_partition"].fillna("<NA>").unique()
                    color_values_mapping = dict(
                        zip(df_haps["_partition"], df_haps[color_column])
                    )
                    color_values_mapping["<NA>"] = "black"
                    color_values_display = [
                        color_values_mapping[c] for c in color_values
                    ]

                # Handle mapping/dictionary case
                elif isinstance(color, Mapping):
                    # For mapping case, we need to create a new column based on the mapping
                    # Initialize with None
                    df_haps["_partition"] = None

                    # Apply each query in the mapping to create the _partition column
                    for label, query in color.items():
                        # Validate and apply the query to matching rows
                        validate_query(query)
                        mask = df_haps.eval(query)
                        df_haps.loc[mask, "_partition"] = label

                    # Clean up the _partition column to avoid issues with special characters
                    if df_haps["_partition"].notna().any():
                        df_haps["_partition"] = df_haps["_partition"].str.replace(
                            r"\W", "", regex=True
                        )

                    # extract all unique values of the color column
                    color_values = df_haps["_partition"].fillna("<NA>").unique()
                    # For mapping case, use _partition values directly as they're already the labels
                    color_values_mapping = dict(
                        zip(df_haps["_partition"], df_haps["_partition"])
                    )
                    color_values_mapping["<NA>"] = "black"
                    color_values_display = [
                        color_values_mapping[c] for c in color_values
                    ]
                else:
                    # Invalid type
                    raise TypeError(
                        f"Expected color parameter to be a string or mapping, got {type(color).__name__}"
                    )

                # count color values for each distinct haplotype (same for both string and mapping cases)
                ht_color_counts = [
                    df_haps.iloc[list(s)]["_partition"].value_counts().to_dict()
                    for s in ht_distinct_sets
                ]

                # Set up colors (same for both string and mapping cases)
                (
                    color_prepped,
                    color_discrete_map_prepped,
                    category_orders_prepped,
                ) = self._setup_sample_colors_plotly(
                    data=df_haps,
                    color="_partition",
                    color_discrete_map=color_discrete_map,
                    color_discrete_sequence=color_discrete_sequence,
                    category_orders=category_orders,
                )
                del color_discrete_map
                del color_discrete_sequence
                del category_orders
                color_discrete_map_display = {
                    color_values_mapping[v]: c
                    for v, c in color_discrete_map_prepped.items()
                }

        debug("construct graph")
        anon_width = np.sqrt(0.3 * node_size_factor)
        graph_nodes, graph_edges = _mjn_graph(
            ht_distinct=ht_distinct,
            ht_distinct_mjn=ht_distinct_mjn,
            ht_counts=ht_counts,
            ht_color_counts=ht_color_counts,
            color="_partition" if color is not None else None,
            color_values=color_values,
            edges=edges,
            alt_edges=alt_edges,
            node_size_factor=node_size_factor,
            anon_width=anon_width,
        )

        debug("prepare graph data for cytoscape")
        elements = [{"data": n} for n in graph_nodes] + [
            {"data": e} for e in graph_edges
        ]

        debug("define node style")
        node_style = {
            "width": "data(width)",
            "height": "data(width)",
            "pie-size": "100%",
        }
        if color and color_discrete_map_prepped is not None:
            # here are the styles which control the display of nodes as pie
            # charts
            for i, (v, c) in enumerate(color_discrete_map_prepped.items()):
                node_style[f"pie-{i + 1}-background-color"] = c
                node_style[
                    f"pie-{i + 1}-background-size"
                ] = f"mapData({v}, 0, 100, 0, 100)"
        node_stylesheet = {
            "selector": "node",
            "style": node_style,
        }
        debug(node_stylesheet)

        debug("define edge style")
        edge_stylesheet = {
            "selector": "edge",
            "style": {"curve-style": "bezier", "width": 2, "opacity": 0.5},
        }

        debug("define style for selected node")
        selected_stylesheet = {
            "selector": ":selected",
            "style": {
                "border-width": "3px",
                "border-style": "solid",
                "border-color": "black",
            },
        }

        debug("create figure legend")
        if color is not None:
            legend_fig = _plotly_discrete_legend(
                color="_partition",  # Changed from color=color
                color_values=color_values_display,
                color_discrete_map=color_discrete_map_display,
                category_orders=category_orders_prepped,
            )
            legend_component = dcc.Graph(
                id="legend",
                figure=legend_fig,
                config=dict(
                    displayModeBar=False,
                ),
            )
        else:
            legend_component = html.Div()

        debug("define cytoscape component")
        if layout_params is None:
            graph_layout_params = dict()
        else:
            graph_layout_params = dict(**layout_params)
        graph_layout_params["name"] = layout
        graph_layout_params.setdefault("padding", 10)
        graph_layout_params.setdefault("animate", False)

        cytoscape_component = cyto.Cytoscape(
            id="cytoscape",
            elements=elements,
            layout=graph_layout_params,
            stylesheet=[
                node_stylesheet,
                edge_stylesheet,
                selected_stylesheet,
            ],
            style={
                # width and height needed to get cytoscape component to display
                "width": "100%",
                "height": "100%",
                "background-color": "white",
            },
            # enable selecting multiple nodes with shift click and drag
            boxSelectionEnabled=True,
            # prevent accidentally zooming out to oblivion
            minZoom=0.1,
            # lower scroll wheel zoom sensitivity to prevent accidental zooming when trying to navigate large graphs
            wheelSensitivity=0.1,
        )

        debug("create dash app")
        app = Dash(
            "dash-cytoscape-network",
            # this stylesheet is used to provide support for a rows and columns
            # layout of the components
            external_stylesheets=["https://codepen.io/chriddyp/pen/bWLwgP.css"],
        )
        # it's generally faster to serve script files from CDN
        app.scripts.config.serve_locally = serve_scripts_locally
        app.layout = html.Div(
            [
                html.Div(
                    cytoscape_component,
                    className="nine columns",
                    style={
                        # required to get cytoscape component to show ...
                        # reduce to prevent scroll overflow
                        "height": f"{height - 50}px",
                        "border": "1px solid black",
                    },
                ),
                html.Div(
                    legend_component,
                    className="three columns",
                    style={
                        "height": f"{height - 50}px",
                    },
                ),
                html.Div(id="output"),
            ],
        )

        debug(
            "define a callback function to display information about the selected node"
        )

        @app.callback(Output("output", "children"), Input("cytoscape", "tapNodeData"))
        def display_tap_node_data(data):
            if data is None:
                return "Click or tap a node for more information."
            else:
                n = data["count"]
                text = f"No. haplotypes: {n}"
                selected_color_data = {
                    color_v_display: int(data.get(color_v, 0) * n / 100)
                    for color_v, color_v_display in zip(
                        color_values, color_values_display
                    )
                }
                selected_color_data = sorted(
                    selected_color_data.items(), key=lambda item: item[1], reverse=True
                )
                color_texts = [
                    f"{color_v}: {color_n}"
                    for color_v, color_n in selected_color_data
                    if color_n > 0
                ]
                if color_texts:
                    color_texts = "; ".join(color_texts)
                    text += f" ({color_texts})"
                return text

        debug("set up run parameters")
        # workaround weird mypy bug here
        run_params: Dict[str, Any] = dict()
        if height is not None:
            run_params["jupyter_height"] = height
        if width is not None:
            run_params["jupyter_width"] = width
        if server_mode is not None:
            run_params["jupyter_mode"] = server_mode
        if server_port is not None:
            run_params["port"] = server_port

        debug("launch the dash app")
        app.run(**run_params)
