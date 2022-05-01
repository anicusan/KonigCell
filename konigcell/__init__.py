#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.05.2021


from    .pixels         import  Pixels
from    .voxels         import  Voxels

from    .               import  kc2d
from    .               import  kc3d

from    .compute2d      import  dynamic2d, static2d
from    .compute2d      import  dynamic_prob2d, static_prob2d

from    .compute3d      import  dynamic3d, static3d
from    .compute3d      import  dynamic_prob3d, static_prob3d


# Import package version
from    .__version__    import  __version__


# Mode enum: kc2d_mode and kc3d_mode
from    .               import  mode
from    .mode           import  RATIO, INTERSECTION, PARTICLE, ONE




def format_fig(
    fig,
    size = 15,
    font = "Computer Modern",
    template = "plotly_white",
):
    '''Format a Plotly figure to a consistent theme for the Nature
    Computational Science journal.
    '''

    # LaTeX font
    fig.update_layout(
        font_family = font,
        font_size = size,
        title_font_family = font,
        title_font_size = size,
    )

    for an in fig.layout.annotations:
        an["font"]["size"] = size

    fig.update_xaxes(title_font_family = font, title_font_size = size)
    fig.update_yaxes(title_font_family = font, title_font_size = size)
    fig.update_layout(template = template)


def create_fig(nrows = 1, ncols = 1, subplot_titles = [], **kwargs):
    '''Return a ``plotly.graph_objs.Figure`` pre-formatted with equalised
    axes and white theming.
    '''

    from plotly.subplots import make_subplots

    subplot_titles = list(subplot_titles) + [""] * (
        nrows * ncols - len(subplot_titles)
    )
    fig = make_subplots(
        nrows,
        ncols,
        subplot_titles = subplot_titles,
        **kwargs,
    )

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j + 1
            yaxis = f"yaxis{index}" if index != 1 else "yaxis"
            xax = f"x{index}" if index != 1 else "x"

            fig.layout[yaxis].update(scaleanchor = xax, scaleratio = 1)

    format_fig(fig)
    return fig




__author__ = "Andrei Leonard Nicusan"
__email__ = "a.l.nicusan@bham.ac.uk"
__license__ = "MIT"
__status__ = "Beta"
