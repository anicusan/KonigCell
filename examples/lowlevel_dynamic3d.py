#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : lowlevel_dynamic3d.py
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.08.2021


import time
import numpy as np
import konigcell as kc
import plotly.graph_objs as go


# Define pixel grid resolution and physical dimensions
resolution = (100, 100, 100)
xlim = [-100, 100]
ylim = [-100, 100]
zlim = [-100, 100]

grid = np.zeros(resolution)
voxels = kc.Voxels(grid, xlim=xlim, ylim=ylim, zlim=zlim)


# Generate random 3D particles / trajectories
np.random.seed(0)

num_particles = 1000
radii_range = [0.1, 5]


def generate(shape, vmin=0, vmax=1):
    return vmin + np.random.random(shape) * (vmax - vmin)


positions = generate(
    (num_particles, 3),
    xlim[0] + radii_range[1],       # Ensure all particles are inside box for
    xlim[1] - radii_range[1],       # the analytical area comparison
)
radii = generate(num_particles, radii_range[0], radii_range[1])


# Pixellise dynamic particles
start = time.time()
kc.kc3d.dynamic3d(
    voxels.voxels,
    voxels.xlim,
    voxels.ylim,
    voxels.zlim,
    positions,
    kc.INTERSECTION,
    radii = radii,
)
end = time.time()
print(f"Voxellised in {end - start} s.")

# Compare pixellised area vs. analytical value
volume_pixellised = float(voxels.sum())
volume_analytical = 0.5 * 4 / 3 * np.pi * radii[0] ** 3
for i in range(num_particles - 1):
    distance = positions[i] - positions[i + 1]
    volume_analytical += np.pi / 3 * np.linalg.norm(distance) * (
        radii[i] ** 2 + radii[i] * radii[i + 1] + radii[i + 1] ** 2
    )

volume_analytical += 0.5 * 4 / 3 * np.pi * radii[num_particles - 1] ** 3
error = 1 - volume_pixellised / volume_analytical

print(f"Voxellised volume = {volume_pixellised}")
print(f"Analytical volume = {volume_analytical}")
print(f"Error =             {error * 100:6.6f}%")

# Matplotlib plotting
fig, ax = voxels.plot()
fig.show()

# Plotly plotting
fig2 = go.Figure()

# fig2.add_traces(voxels.cubes_traces())
fig2.add_trace(voxels.scatter_trace())

kc.format_fig(fig2)
fig2.show()
