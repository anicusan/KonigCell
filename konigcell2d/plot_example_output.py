'''Plot grid of particles pixellised using konigcell2d. You first need to
compile `example.c` and `konigcell2d.c` and redirect the printed grid to a
file, whose path you should give to this Python script.

For example:
  $> clang example.c konigcell2d.c -lm -O3
  $> ./a.out > pixels.csv
  $> python plot_example_output.py pixels.csv
'''

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
