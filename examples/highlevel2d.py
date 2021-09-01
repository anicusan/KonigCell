#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : highlevel2d.py
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 01.09.2021


import numpy as np
import konigcell as kc
import plotly.graph_objs as go


def generate(shape, vmin=0, vmax=1):
    '''Return an array of `shape` with random values between `vmin` and `vmax`.
    '''
    return vmin + np.random.random(shape) * (vmax - vmin)


# -----------------------------------------------------------------------------
# Generate random 2D particles / trajectory
np.random.seed(0)

num_particles = 100
positions = generate((num_particles, 2), -10, 10)
radii = generate(num_particles, 0.1, 2)


# -----------------------------------------------------------------------------
# Pixellise the 2D particle's moving / dynamic trajectory
pixels = kc.dynamic2d(positions, kc.INTERSECTION, radii = radii,
                      resolution = (200, 200))

# Plot pixels
fig = go.Figure()
fig.add_trace(pixels.heatmap_trace())
kc.format_fig(fig)
fig.show()


# -----------------------------------------------------------------------------
# Pixellise the 2D particles' individual / static locations
pixels2 = kc.static2d(positions, kc.INTERSECTION, radii = radii,
                      resolution = (200, 200))

# Plot pixels
fig2 = go.Figure()
fig2.add_trace(pixels2.heatmap_trace())
kc.format_fig(fig2)
fig2.show()
