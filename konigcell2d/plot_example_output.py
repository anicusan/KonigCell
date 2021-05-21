import sys
import numpy as np
import konigcell as kc
import plotly.graph_objs as go


xmin, xmax, ymin, ymax = np.loadtxt(sys.argv[1], max_rows = 1)
grid = np.loadtxt(sys.argv[1], skiprows = 1)

pixels = kc.Pixels(grid, (xmin, xmax), (ymin, ymax))

fig = go.Figure()
fig.add_trace(pixels.heatmap_trace())
fig.show()
