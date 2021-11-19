#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : highlevel2d.py
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 01.09.2021


import numpy as np
import konigcell as kc
import plotly.graph_objs as go


# -----------------------------------------------------------------------------
# Generate random 2D particles / trajectory
rng = np.random.default_rng(0)

num_particles = 100
positions = rng.uniform(-10, 10, (num_particles, 2))
radii = rng.uniform(0.1, 2, num_particles)


# -----------------------------------------------------------------------------
# Pixellise the 2D particle's moving / dynamic trajectory
pixels = kc.dynamic2d(positions, kc.INTERSECTION, radii = radii,
                      resolution = (200, 200))
print(pixels)

# Plot pixels
fig = go.Figure()
fig.add_trace(pixels.heatmap_trace())
kc.format_fig(fig)
fig.show()


# -----------------------------------------------------------------------------
# Pixellise the 2D particles' individual / static locations
pixels2 = kc.static2d(positions, kc.INTERSECTION, radii = radii,
                      resolution = (200, 200))
print(pixels2)

# Plot pixels
fig2 = go.Figure()
fig2.add_trace(pixels2.heatmap_trace())
kc.format_fig(fig2)
fig2.show()
