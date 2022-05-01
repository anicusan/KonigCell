Rasterization Modes
===================


::

    import numpy as np
    import konigcell as kc

    # Generate accelearting trajectory to rasterize
    n = 1000
    x = np.linspace(3, 1, n) * np.cos(np.linspace(0, 100, n))
    y = np.linspace(2, 0, n) * np.sin(np.linspace(0, 100, n))

    positions = np.vstack((x, y)).T
    radii = 0.2
    velocities = np.linspace(0, 1, n - 1)

    # Pixellise the moving particle using different rasterization modes
    modes = [kc.ONE, kc.PARTICLE, kc.INTERSECTION, kc.RATIO]

    pixels = []
    for mode in modes:
        pix = kc.dynamic2d(
            positions,
            mode,
            radii = radii,
            values = velocities,
            resolution = (500, 500),
        )

        pixels.append(pix)

    # Show a Plotly heatmap of the pixels
    fig = kc.create_fig(nrows = 2, ncols = 2, subplot_titles = [
        "ONE", "PARTICLE", "INTERSECTION", "RATIO",
    ])

    fig.add_trace(pixels[0].heatmap_trace(), row = 1, col = 1)
    fig.add_trace(pixels[1].heatmap_trace(), row = 1, col = 2)
    fig.add_trace(pixels[2].heatmap_trace(), row = 2, col = 1)
    fig.add_trace(pixels[3].heatmap_trace(), row = 2, col = 2)

    fig.show()


.. image:: raster_modes.png



