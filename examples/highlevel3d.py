#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : dynamic2d.py
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 01.09.2021


import numpy as np
import konigcell as kc
import plotly.graph_objs as go


# -----------------------------------------------------------------------------
# Generate random 3D particles / trajectory
rng = np.random.default_rng(0)

num_particles = 100
positions = rng.uniform(-10, 10, (num_particles, 3))
radii = rng.uniform(0.1, 2, num_particles)


# -----------------------------------------------------------------------------
# Voxellise the 3D particle's moving / dynamic trajectory
voxels = kc.dynamic3d(positions, kc.INTERSECTION, radii = radii,
                      resolution = (50, 50, 50))

# Plot voxels; only use `voxels.cubes_traces()` if you have very few voxels
fig = go.Figure()

fig.add_trace(voxels.scatter_trace())
# fig.add_traces(voxels.cubes_traces()).show()

kc.format_fig(fig)
fig.show()

# Interactive volumetric plot with PyVista
# voxels.plot_volumetric().show()


# -----------------------------------------------------------------------------
# Voxellise the 3D particles' individual / static locations
voxels2 = kc.static3d(positions, kc.INTERSECTION, radii = radii,
                      resolution = (50, 50, 50))

# Plot voxels; only use `voxels.cubes_traces()` if you have very few voxels
fig2 = go.Figure()

fig2.add_trace(voxels2.scatter_trace())
# fig2.add_traces(voxels.cubes_traces()).show()

kc.format_fig(fig2)
fig2.show()

# Interactive volumetric plot with PyVista
# voxels2.plot_volumetric().show()
