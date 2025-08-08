import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy.cluster.hierarchy as sch

def plot_dendrogram(
    dist,
    linkage_method="ward",
    count_sort=False,
    distance_sort=False,
    render_mode="svg",
    width=800,
    height=600,
    title="Dendrogram with Optional Volcanic Heatmap",
    line_width=2,
    line_color="black",
    marker_size=8,
    leaf_data=None,
    leaf_hover_name=None,
    leaf_hover_data=None,
    leaf_color=None,
    leaf_symbol=None,
    leaf_y=-0.5,
    leaf_color_discrete_map=None,
    leaf_category_orders=None,
    template="plotly_white",
    y_axis_title="Distance",
    y_axis_buffer=0.5,
    add_heatmap=False,  # New parameter to toggle heatmap
    heatmap_data=None  # Mutation data matrix for heatmap
):
    """
    Plots a hierarchical dendrogram with an optional volcanic heatmap overlay.

    Parameters:
    - dist: Distance matrix for clustering.
    - linkage_method: Linkage method for hierarchical clustering.
    - count_sort: Sort clusters by count.
    - distance_sort: Sort clusters by distance.
    - render_mode: Rendering mode for Plotly.
    - width, height: Figure dimensions.
    - title: Title of the figure.
    - line_width, line_color: Style for dendrogram lines.
    - marker_size: Size of leaf markers.
    - leaf_data: Data for leaf nodes.
    - leaf_hover_name, leaf_hover_data: Hover labels.
    - leaf_color, leaf_symbol: Leaf node aesthetics.
    - leaf_y: Position of leaf nodes.
    - template: Plotly figure template.
    - y_axis_title: Label for the y-axis.
    - y_axis_buffer: Buffer for y-axis scaling.
    - add_heatmap: Boolean flag to add a volcanic heatmap.
    - heatmap_data: Mutation data for heatmap (must align with dendrogram order).
    """
    # Hierarchical clustering
    Z = sch.linkage(dist, method=linkage_method)
    dend = sch.dendrogram(Z, count_sort=count_sort, distance_sort=distance_sort, no_plot=True)

    # Extract dendrogram coordinates
    icoord, dcoord = dend["icoord"], dend["dcoord"]
    line_segments_x, line_segments_y = [], []
    for ik, dk in zip(icoord, dcoord):
        line_segments_x += ik + [None]
        line_segments_y += dk + [None]
    df_line_segments = pd.DataFrame({"x": line_segments_x, "y": line_segments_y})
    df_line_segments["x"] = (df_line_segments["x"] - 5) / 10

    # Plot dendrogram lines
    fig = px.line(df_line_segments, x="x", y="y", render_mode=render_mode, template=template)

    # Reorder leaf data to align with dendrogram
    leaves = dend["leaves"]
    n_leaves = len(leaves)
    leaf_data = leaf_data.iloc[leaves] if leaf_data is not None else None

    # Add scatter plot for leaf nodes
    fig.add_traces(
        list(
            px.scatter(
                data_frame=leaf_data,
                x=np.arange(n_leaves),
                y=np.repeat(leaf_y, n_leaves),
                color=leaf_color,
                symbol=leaf_symbol,
                render_mode=render_mode,
                hover_name=leaf_hover_name,
                hover_data=leaf_hover_data,
                template=template,
                color_discrete_map=leaf_color_discrete_map,
                category_orders=leaf_category_orders,
            ).select_traces()
        )
    )

    # Style dendrogram lines and markers
    fig.update_traces(line=dict(width=line_width, color=line_color), marker=dict(size=marker_size))

    # Add heatmap if enabled
    if add_heatmap and heatmap_data is not None:
        heatmap_data = heatmap_data.iloc[leaves, :]  # Align heatmap with dendrogram order
        heatmap_fig = go.Heatmap(
            z=heatmap_data.values,
            x=list(range(n_leaves)),
            y=heatmap_data.columns,
            colorscale="hot",  # Volcanic color scheme
            showscale=True,
        )
        fig.add_trace(heatmap_fig)

    # Style layout
    fig.update_layout(
        width=width,
        height=height,
        title=title,
        hovermode="closest",
        yaxis_title=y_axis_title,
        showlegend=True,
    )

    # Style axes
    fig.update_xaxes(showgrid=False, showticklabels=False, range=(-2, n_leaves + 2))
    fig.update_yaxes(range=(leaf_y - y_axis_buffer, np.max(dcoord) + y_axis_buffer))

    return fig, leaf_data
