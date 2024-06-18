import numpy as np
import pandas as pd
import plotly.express as px
import scipy.cluster.hierarchy as sch


def plot_dendrogram(
    dist,
    linkage_method,
    count_sort,
    distance_sort,
    render_mode,
    width,
    height,
    title,
    line_width,
    line_color,
    marker_size,
    leaf_data,
    leaf_hover_name,
    leaf_hover_data,
    leaf_color,
    leaf_symbol,
    leaf_y,
    leaf_color_discrete_map,
    leaf_category_orders,
    template,
    y_axis_title,
    y_axis_buffer,
):
    # Hierarchical clustering.
    Z = sch.linkage(dist, method=linkage_method)

    # Compute the dendrogram but don't plot it.
    dend = sch.dendrogram(
        Z,
        count_sort=count_sort,
        distance_sort=distance_sort,
        no_plot=True,
    )

    # Compile the line coordinates into a single dataframe.
    icoord = dend["icoord"]
    dcoord = dend["dcoord"]
    line_segments_x = []
    line_segments_y = []
    for ik, dk in zip(icoord, dcoord):
        # Adding None here breaks up the lines.
        line_segments_x += ik + [None]
        line_segments_y += dk + [None]
    df_line_segments = pd.DataFrame({"x": line_segments_x, "y": line_segments_y})

    # Convert X coordinates to haplotype indices (scipy multiplies coordinates by 10).
    df_line_segments["x"] = (df_line_segments["x"] - 5) / 10

    # Plot the lines.
    fig = px.line(
        df_line_segments,
        x="x",
        y="y",
        render_mode=render_mode,
        template=template,
    )

    # Reorder leaf data to align with dendrogram.
    leaves = dend["leaves"]
    n_leaves = len(leaves)
    leaf_data = leaf_data.iloc[leaves]

    # Add scatter plot to draw the leaves.
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

    # Style the lines and markers.
    line_props = dict(
        width=line_width,
        color=line_color,
    )
    marker_props = dict(
        size=marker_size,
    )
    fig.update_traces(line=line_props, marker=marker_props)

    # Style the figure.
    fig.update_layout(
        width=width,
        height=height,
        title=title,
        autosize=True,
        hovermode="closest",
        # I cannot get the xaxis title to appear below the plot, and when
        # it's above the plot it often overlaps the title, so hiding it
        # for now.
        xaxis_title=None,
        yaxis_title=y_axis_title,
        showlegend=True,
    )

    # Style axes.
    fig.update_xaxes(
        mirror=False,
        showgrid=False,
        showline=False,
        showticklabels=False,
        ticks="",
        range=(-2, n_leaves + 2),
    )
    fig.update_yaxes(
        mirror=False,
        showgrid=False,
        showline=False,
        showticklabels=True,
        ticks="outside",
        range=(leaf_y - y_axis_buffer, np.max(dcoord) + y_axis_buffer),
    )

    return fig, leaf_data
