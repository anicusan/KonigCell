#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_pixels_voxels.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 22.11.2021


'''Integration tests to ensure the `konigcell` base classes behave correctly
and offer consistent interfaces.
'''


import pytest
import numpy as np

import konigcell as kc


def test_pixels():
    pixels_raw = np.arange(64).reshape(8, 8)
    xlim = [10, 20]
    ylim = [-10, 0]

    pixels = kc.Pixels(pixels_raw, xlim, ylim)
    print(pixels)

    assert float(pixels.pixels.sum()) == float(pixels_raw.sum())

    # Testing methods
    kc.Pixels.zeros((5, 5), xlim, ylim)

    pixels.copy()
    pixels.copy(pixels_raw)
    pixels.copy(xlim = [20, 30])

    pixels.attrs["a1"] = 12345
    pixels.xlim[0] = 15

    pixels.to_physical([1, 1])
    pixels.from_physical([10, -10])

    # Test dynamic attribute creation
    pixels.lower
    pixels.upper
    pixels.pixel_grids
    pixels.pixel_size
    pixels.pixels
    pixels.xlim
    pixels.ylim

    # Test plotting functions - only locally, as they are optional deps
    # pixels.heatmap_trace()
    # pixels.plot()


def test_voxels():
    voxels_raw = np.arange(64).reshape(4, 4, 4)
    xlim = [10, 20]
    ylim = [-10, 0]
    zlim = [15, 40]

    voxels = kc.Voxels(voxels_raw, xlim, ylim, zlim)
    print(voxels)

    assert float(voxels.voxels.sum()) == float(voxels_raw.sum())

    # Testing methods
    kc.Voxels.zeros((5, 5, 5), xlim, ylim, zlim)

    voxels.copy()
    voxels.copy(voxels_raw)
    voxels.copy(xlim = [20, 30])

    voxels.attrs["a1"] = 12345
    voxels.xlim[0] = 15

    voxels.to_physical([1, 1, 1])
    voxels.from_physical([10, -10, 15])

    # Test dynamic attribute creation
    voxels.lower
    voxels.upper
    voxels.voxel_grids
    voxels.voxel_size
    voxels.voxels
    voxels.xlim
    voxels.ylim

    # Test plotting functions - only locally, as they are optional deps
    # voxels.heatmap_trace(ix = 2)
    # voxels.plot()
    # voxels.vtk()
    # voxels.plot_volumetric()
    # voxels.cube_trace((2, 2, 2))
    # voxels.cubes_traces()
    # voxels.scatter_trace()
