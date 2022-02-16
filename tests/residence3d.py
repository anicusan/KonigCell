#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : residence.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 10.02.2022


import numpy as np
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
voxels = kc.Voxels.zeros(
    (100, 100, 100),
    [-2, 8],
    [-2, 2 + 2 * len(trajs)],
    [-2, 2 + 2 * len(trajs)],
)


# Compute residence time distribution
traces = []
for i, traj in enumerate(trajs):
    traj[:, 2] = i * 2

    t = traj[:, 0]
    dt = t[1:] - t[:-1]

    kc.dynamic3d(
        traj[:, [1, 2, 2]],
        kc.RATIO,
        values = dt,
        radii = radii,
        voxels = voxels,
        max_workers = 1,
        verbose = False,
    )

    traces.append(go.Scatter3d(
        x = traj[:, 1],
        y = traj[:, 2],
        z = traj[:, 2],
        mode = "markers+lines",
        line_color = "red",
    ))

fig = go.Figure()
fig.add_trace(voxels.scatter_trace())
fig.add_traces(traces)

kc.format_fig(fig)
fig.update_scenes(
    aspectmode = "manual",
    aspectratio = dict(x = 1, y = 1, z = 1),
)
fig.show()
