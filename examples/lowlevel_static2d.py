#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : lowlevel_static.py
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.05.2021


import time
import numpy as np
import konigcell as kc
import plotly.graph_objs as go


# Define pixel grid resolution and physical dimensions
resolution = (1980, 1080)
xlim = [-100, 100]
ylim = [-100, 100]

grid = np.zeros(resolution)
pixels = kc.Pixels(grid, xlim=xlim, ylim=ylim)


# Generate random 2D particles / trajectories
num_particles = 1000
radii_range = [0.1, 5]


def generate(shape, vmin=0, vmax=1):
    return vmin + np.random.random(shape) * (vmax - vmin)


positions = generate(
    (num_particles, 2),
    xlim[0] + radii_range[1],       # Ensure all particles are inside box for
    xlim[1] - radii_range[1],       # the analytical area comparison
)
radii = generate(num_particles, radii_range[0], radii_range[1])

# Pixellise static particles
start = time.time()
kc.kc2d.static2d(
    pixels.pixels,
    pixels.xlim,
    pixels.ylim,
    positions,
    kc.INTERSECTION,
    radii = radii,
)
end = time.time()
print(f"Pixellised in {end - start} s.")

# Compare pixellised area vs. analytical value
area_pixellised = pixels.pixels.sum()
area_analytical = (np.pi * radii * radii).sum()
error = 1 - area_pixellised / area_analytical

print(f"Pixellised area = {area_pixellised}")
print(f"Analytical area = {area_analytical}")
print(f"Error =           {error * 100:4.4f}%")

# Matplotlib plotting
fig, ax = pixels.plot()
fig.show()

# Plotly plotting
fig2 = go.Figure()
fig2.add_trace(pixels.heatmap_trace())
kc.format_fig(fig2)
fig2.show()
