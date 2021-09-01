import sys
import numpy as np
import konigcell as kc
import plotly.graph_objs as go


xdim, ydim, zdim = np.loadtxt(sys.argv[1], max_rows = 1, dtype = int)
xmin, xmax, ymin, ymax, zmin, zmax = np.loadtxt(sys.argv[1], skiprows = 1,
                                                max_rows = 1)

grid = np.loadtxt(sys.argv[1], skiprows = 2).reshape(xdim, ydim, zdim)
voxels = kc.Voxels(grid, (xmin, xmax), (ymin, ymax), (zmin, zmax))

fig = go.Figure()

# Use `cubes_traces` for few voxels (< 20x20x20)
# fig.add_traces(voxels.cubes_traces())
fig.add_trace(voxels.scatter_trace())

fig.update_scenes(
    xaxis_range = [xmin, xmax],
    yaxis_range = [ymin, ymax],
    zaxis_range = [zmin, zmax],
)
fig.show()
