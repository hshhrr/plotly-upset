import numpy as np

import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .util import *


def get_styled_figure(
    fig = None,
    height: int = None,
    width: int = None,
    plot_range_x: int = None,
    sets: list = None,
):
    # <- ROW 1, COL 2 ->
    fig.update_xaxes(
        # type="linear", # <-
        side='bottom',
        showline=True,
        showticklabels=False,
        linecolor='#000000',
        ticks='outside',
        tickcolor='#000000',
        dtick=1,
        row=1, col=2
    )

    fig.update_yaxes(
        # type="log", # <-
        side='left',
        showgrid=True,
        showline=True,
        showticklabels=True,
        title="Intersection Size",
        title_standoff=5,
        title_font_color='#000000',
        linecolor='#000000',
        gridcolor='#E0E0E0',
        ticks='outside',
        tickcolor='#000000',
        row=1, col=2
    )

    # <- ROW 2, COL 1 ->
    fig.update_xaxes(
        # type="log", # <-
        side='top',
        autorange='reversed',
        showgrid=True,
        showline=True,
        showticklabels=True,
        title="Set Size",
        title_standoff=5,
        title_font_color='#000000',
        gridcolor='#E0E0E0',
        linecolor='#000000',
        ticks='outside',
        tickcolor='#000000',
        # tickangle=-90, # <-
        row=2, col=1
    )

    fig.update_yaxes(
        side='right',
        showline=True,
        showticklabels=False,
        linecolor='#000000',
        ticks='outside',
        tickcolor='#000000',
        row=2, col=1
    )
    
    # <- ROW 2 COL 2 ->
    fig.update_xaxes(
        side='bottom',
        showline=True,
        showgrid=False,
        showticklabels=False,
        title_standoff=5,
        dtick=1,
        row=2, col=2
    )

    fig.update_yaxes(
        side='left',
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='#000000',
        ticks='outside',
        tickvals=np.arange(0, len(sets)),
        ticktext=sets,
        tickcolor='#000000',
        row=2, col=2
    )

    fig.update_xaxes(
        range=[0. - 0.5, plot_range_x - 0.5],
        # range=[0. - 0.5, (2 ** len(sets)) - 0.5],
        col=2
    )

    fig.update_yaxes(
        range=[0. - 0.5, len(sets) - 0.5],
        row=2
    )

    # <- Common ->
    fig.update_xaxes(
        zeroline=False,
    )

    fig.update_yaxes(
        zeroline=False,
    )

    fig.update_layout(
        uniformtext_minsize=12,
        uniformtext_mode='show',
        barmode='group',
        bargap=0.2, bargroupgap=0.0,
        margin=dict(l=40, r=40, t=40, b=40),
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1.0,
            xanchor='left',
            x=0
        ),
        # showlegend=True,
        height=height,
        width=width,
        paper_bgcolor="white",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        hovermode='closest',
        font_color="black",
    )

    return fig


def plot_upset(
    dataframes: list,
    legendgroups: list,
    set_names: list = None,
    marker_colors: list = None,
    exclude_zeros: bool = False,
    sorted_x: str = None,
    sorted_y: str = None,
    row_heights: list = [0.7, 0.3],
    column_widths: list = [0.4, 0.6],
    vertical_spacing: float = 0.05,
    horizontal_spacing: float = 0.125,
    marker_size: int = 12,
    height: int = 400,
    width: int = 640,
):
    # Error Handling
    df_columns = [df.columns for df in dataframes]
    lengths = [len(x) for x in df_columns]
    df = dataframes[0]

    if len(np.unique(lengths)) != 1:
        raise Exception("DataFrames don't share same number of columns.")
    elif len(np.unique(df_columns)) != np.unique(lengths):
        raise Exception("DataFrames don't share same columns.")
    elif len(legendgroups) != len(dataframes):
        raise Exception("Number of DataFrames and Number of Legend Groups don't match.")
    elif set_names is not None and len(set_names) != len(df.columns):
        raise Exception("Number of DataFrame Columns and Number of Set Names don't match.")
    elif marker_colors is not None and len(marker_colors) != len(dataframes):
        raise Exception("Number of DataFrames and Number of Marker Colors don't match.")
    elif sorted_x is not None and (sorted_x.lower() not in ["a", "d", "ascending", "descending"]):
        raise Exception("Unknown sorting order.")
    elif sorted_y is not None and (sorted_y.lower() not in ["a", "d", "ascending", "descending"]):
        raise Exception("Unknown sorting order.")
    elif (sorted_x is not None or sorted_x is not None) and len(dataframes) > 1:
        raise Exception("Sorting isn't available for multiple DataFrames.")
    elif exclude_zeros is True and len(dataframes) > 1:
        raise Exception("Zero value exclusion isn't available for multiple DataFrames.")
    elif len(row_heights) != 2 or sum(row_heights) != 1.0:
        raise Exception("Invalid Row Heights.")
    elif len(column_widths) != 2 or sum(column_widths) != 1.0:
        raise Exception("Invalid Column Widths.")
    elif vertical_spacing > 1.0:
        raise Exception("Invalid Vertical Spacing.")
    elif horizontal_spacing > 1.0:
        raise Exception("Invalid Horizontal Spacing.")

    if set_names is not None:
        sets = set_names
    else:
        sets = df.columns

    # Plotting
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=row_heights,
        column_widths=column_widths,
        vertical_spacing = vertical_spacing,
        horizontal_spacing = horizontal_spacing,
        shared_xaxes=True,
    )

    string_repr, _ = possible_intersections(len(sets))
    
    for i, df in enumerate(dataframes):
        int_ss = np.array(intersecting_set_size(df))
        ind_ss = np.array(individual_set_size(df))
        t, f, edges = get_nodes_and_edges(n_sets=len(sets))

        if sorted_x is not None:
            a, b, c = int_ss, string_repr, np.arange(0, 2 ** len(sets))
            order = False if sorted_x.lower() == "a" or sorted_x.lower() == "ascending" else True
            sorted_list = sorted(zip(a, b, c), reverse=order)
            transposed = np.array(sorted_list).T

            a, b, c = transposed

            int_ss, string_repr, _sorted_sequence = a.astype(int), b, c.astype(int)

            t, f, edges = get_sorted_nodes_and_edges(
                t=t, f=f, edges=edges,
                sorted_sequence=_sorted_sequence
            )

        if sorted_y is not None:
            a, b = ind_ss, sets
            order = False if sorted_y.lower() == "a" or sorted_x.lower() == "ascending" else True
            sorted_list = sorted(zip(a, b), reverse=order)
            transposed = np.array(sorted_list).T
            a, b = transposed
            ind_ss, sets = a.astype(int), b

        if exclude_zeros:
            nonzero_indices = np.where(int_ss != 0)[0]
            string_repr = np.array(string_repr)[int_ss != 0]
            int_ss = int_ss[int_ss != 0]

            t, f, edges = get_nonzero_nodes_and_edges(
                t=t, f=f, edges=edges,
                nonzero_indices=nonzero_indices
            )

        plot_range_x = len(int_ss)

        # <- ROW 1, COL 2 ->
        fig.add_trace(
            go.Bar(
                x=string_repr,
                y=int_ss,
                text=int_ss,
                texttemplate='%{text:}', textposition='outside', textfont_size=12, textangle=-90, cliponaxis=False,
                legendgroup=legendgroups[i],
                name=legendgroups[i],
                marker_color=marker_colors[i] if marker_colors != None else fig.layout['template']['layout']['colorway'][i],
                showlegend=False,
            ),
            row=1, col=2
        )

        # <- ROW 2, COL 1 ->
        fig.add_trace(
            go.Bar(
                x=ind_ss,
                y=sets,
                text=ind_ss,
                texttemplate='%{text:}', textposition='outside', textfont_size=12, textangle=0, cliponaxis=False,
                legendgroup=legendgroups[i],
                name=legendgroups[i],
                marker_color=marker_colors[i] if marker_colors != None else fig.layout['template']['layout']['colorway'][i],
                orientation='h',
            ),
            row=2, col=1
        )

        # <- ROW 2, COL 2 ->
        # Scatter - True
        xtf = np.concatenate(t[0], axis=None)
        ytf = np.concatenate(t[1], axis=None)
        fig.add_trace(
            go.Scatter(
                x=xtf,
                y=ytf,
                legendgroup='True',
                name='True',
                mode='markers', 
                marker=dict(line_width=0, color='#000000', line_color='#000000', symbol='circle', size=marker_size),
                showlegend=False,
            ),
             row=2, col=2
        )

        # Scatter - False
        xff = np.concatenate(f[0], axis=None)
        yff = np.concatenate(f[1], axis=None)
        fig.add_trace(
            go.Scatter(
                x=xff,
                y=yff,
                legendgroup='False',
                name='False',
                mode='markers', 
                marker=dict(line_width=0, color='#C2C2C2', line_color='#000000', symbol='circle', size=marker_size),
                showlegend=False,
            ),
            row=2, col=2
        )

        # Edges
        for e in edges:
            x, y = np.array(e).T
            fig.add_trace(
                go.Scatter(
                    x=x, y=[y[0], y[1]],
                    legendgroup='True',
                    name='True',
                    mode='lines',
                    line_width=2,
                    line_color='#000000',
                    showlegend=False,
                ),
                row=2, col=2
            )

        for i in range(len(sets)):
            if i % 2 == 0:
                fig.add_hrect(
                    y0=(i - 0.5),
                    y1=(i + 0.5),
                    layer="below",
                    fillcolor="#EBEBEB",
                    line_width=0,
                    row=2, col=2
                )

    fig = get_styled_figure(
        fig=fig,
        height=height,
        width=width,
        plot_range_x=plot_range_x,
        sets=sets,
    )

    return fig
