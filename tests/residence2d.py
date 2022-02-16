#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : residence.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 10.02.2022


import numpy as np
from scipy.interpolate import interp1d
import konigcell as kc

import plotly.graph_objs as go


# Simulate multiple trajectories with various missing bits to check corner
# cases; trajectories' columns: [t, x, y]
traj1 = np.array([
    [0, 0, 0],
    [1, 1, 0],
    [2, 2, 0],
    [3, 3, 0],
    [4, 4, 0],
    [5, 5, 0],
    [6, 6, 0],
])


traj2 = np.array([
    [0, np.nan, np.nan],
    [1, 1, 0],
    [2, 2, 0],
    [3, 3, 0],
    [4, 4, 0],
    [5, 5, 0],
    [6, 6, 0],
])


traj3 = np.array([
    [0, 0, 0],
    [1, np.nan, np.nan],
    [2, 2, 0],
    [3, 3, 0],
    [4, 4, 0],
    [5, 5, 0],
    [6, 6, 0],
])


traj4 = np.array([
    [0, 0, 0],
    [1, 1, 0],
    [2, 2, 0],
    [3, 3, 0],
    [4, 4, 0],
    [5, np.nan, np.nan],
    [6, 6, 0],
])


traj5 = np.array([
    [0, 0, 0],
    [1, 1, 0],
    [2, 2, 0],
    [3, 3, 0],
    [4, 4, 0],
    [5, 5, 0],
    [6, np.nan, np.nan],
])


traj6 = np.array([
    [0, 0, 0],
    [1, 1, 0],
    [2, np.nan, np.nan],
    [3, 3, 0],
    [4, np.nan, np.nan],
    [5, 5, 0],
    [6, 6, 0],
])


traj7 = np.array([
    [0, 0, 0],
    [1, 1, 0],
    [2, 2, 0],
    [3, 3, 0],
    [4, np.nan, np.nan],
    [5, 5, 0],
    [6, np.nan, np.nan],
])


# Pixellise all trajectories onto the same grid, at different heights
trajs = [traj1, traj2, traj3, traj4, traj5, traj6, traj7]
radii = 0.5
pixels = kc.Pixels.zeros((1000, 1000), [-2, 8], [-2, 2 + 2 * len(trajs)])


# Compute residence time distribution
traces = []
for i, traj in enumerate(trajs):
    traj[:, 2] = i * 2

    t = traj[:, 0]
    dt = t[1:] - t[:-1]

    kc.dynamic2d(
        traj[:, [1, 2]],
        kc.RATIO,
        values = dt,
        radii = radii,
        pixels = pixels,
        max_workers = 1,
        verbose = False,
    )

    traces.append(go.Scatter(
        x = traj[:, 1],
        y = traj[:, 2],
        mode = "markers+lines",
        line_color = "red",
    ))

fig = go.Figure()
fig.add_trace(pixels.heatmap_trace())
fig.add_traces(traces)

kc.format_fig(fig)
fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
fig.show()
