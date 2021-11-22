#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : test_pixellise.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 31.08.2021


'''Integration tests to ensure the `konigcell` base classes behave correctly
and offer consistent interfaces.
'''


import pytest

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import konigcell as kc


def generate(shape, vmin=0, vmax=1):
    return vmin + np.random.random(shape) * (vmax - vmin)


def test_dynamic2d():
    # Generate random 3D particles / trajectories
    np.random.seed(0)

    num_particles = 100
    resolution = (20, 20)

    positions = generate((num_particles, 2), -10, 10)
    radii = generate(num_particles, 0.1, 2)

    # Testing different modes of execution return the same results
    pixels = kc.dynamic2d(positions, kc.INTERSECTION, radii = radii,
                          resolution = resolution, max_workers = 1)
    pixels2 = kc.dynamic2d(positions, kc.INTERSECTION, radii = radii,
                           resolution = resolution, verbose = False)
    pixels3 = kc.dynamic2d(positions, kc.INTERSECTION, radii = radii,
                           resolution = resolution,
                           max_workers = num_particles)
    pixels4 = kc.dynamic2d(positions, kc.INTERSECTION, radii = radii,
                           resolution = resolution,
                           executor = ProcessPoolExecutor)

    assert np.isclose(pixels.pixels, pixels2.pixels).all()
    assert np.isclose(pixels.pixels, pixels3.pixels).all()
    assert np.isclose(pixels.pixels, pixels4.pixels).all()

    # Testing different settings work
    kc.dynamic2d(positions, kc.RATIO, radii = radii, resolution = resolution)
    kc.dynamic2d(positions, kc.ONE, pixels = pixels, max_workers = 1)
    kc.dynamic2d(positions, kc.PARTICLE, xlim = [-10, 10], ylim = [-10, 10],
                 resolution = resolution)
    kc.dynamic2d(positions, kc.INTERSECTION, values = 0,
                 resolution = resolution)
    kc.dynamic2d(positions, kc.ONE, radii = 0,
                 resolution = resolution)

    # Testing error cases
    with pytest.raises(ValueError):
        # Missing resolution
        kc.dynamic2d(positions, kc.ONE)

    with pytest.raises(AttributeError):
        # `pixels` has wrong type
        kc.dynamic2d(positions, kc.ONE, pixels = 0)

    with pytest.raises(ValueError):
        # `resolution` has wrong shape
        kc.dynamic2d(positions, kc.ONE, resolution = (1,))

    with pytest.raises(ValueError):
        # `resolution` has wrong numbers
        kc.dynamic2d(positions, kc.ONE, resolution = (1, 1))

    with pytest.raises(ValueError):
        # Wrong number of `values`
        kc.dynamic2d(positions, kc.ONE, np.ones(len(positions)),
                     resolution = (2, 2))


def test_static2d():
    # Generate random 3D particles / trajectories
    np.random.seed(0)

    num_particles = 100
    resolution = (20, 20)

    positions = generate((num_particles, 2), -10, 10)
    radii = generate(num_particles, 0.1, 2)

    # Testing different modes of execution return the same results
    pixels = kc.static2d(positions, kc.INTERSECTION, radii = radii,
                         resolution = resolution, max_workers = 1)
    pixels2 = kc.static2d(positions, kc.INTERSECTION, radii = radii,
                          resolution = resolution, verbose = False)
    pixels3 = kc.static2d(positions, kc.INTERSECTION, radii = radii,
                          resolution = resolution,
                          max_workers = num_particles)
    pixels4 = kc.static2d(positions, kc.INTERSECTION, radii = radii,
                          resolution = resolution,
                          executor = ProcessPoolExecutor)

    assert np.isclose(pixels.pixels, pixels2.pixels).all()
    assert np.isclose(pixels.pixels, pixels3.pixels).all()
    assert np.isclose(pixels.pixels, pixels4.pixels).all()

    # Testing different settings work
    kc.static2d(positions, kc.RATIO, radii = radii, resolution = resolution)
    kc.static2d(positions, kc.ONE, pixels = pixels, max_workers = 1)
    kc.static2d(positions, kc.PARTICLE, xlim = [-10, 10], ylim = [-10, 10],
                resolution = resolution)
    kc.static2d(positions, kc.INTERSECTION, values = 0,
                resolution = resolution)
    kc.static2d(positions, kc.ONE, radii = 0,
                resolution = resolution)

    # Testing error cases
    with pytest.raises(ValueError):
        # Missing resolution
        kc.static2d(positions, kc.ONE)

    with pytest.raises(AttributeError):
        # `pixels` has wrong type
        kc.static2d(positions, kc.ONE, pixels = 0)

    with pytest.raises(ValueError):
        # `resolution` has wrong shape
        kc.static2d(positions, kc.ONE, resolution = (1,))

    with pytest.raises(ValueError):
        # `resolution` has wrong numbers
        kc.static2d(positions, kc.ONE, resolution = (1, 1))

    with pytest.raises(ValueError):
        # Wrong number of `values`
        kc.static_prob2d(positions, np.ones(len(positions) - 1),
                         resolution = (2, 2))


def test_dynamic_prob2d():
    # Generate random 3D particles / trajectories
    np.random.seed(0)

    num_particles = 100
    resolution = (20, 20)

    positions = generate((num_particles, 2), -10, 10)
    values = generate(num_particles - 1, 1, 2)
    radii = generate(num_particles, 0.1, 2)

    pixels = kc.dynamic_prob2d(positions, values, radii = radii,
                               resolution = resolution)

    # Testing different settings work
    kc.dynamic_prob2d(positions, values, radii = radii,
                      resolution = resolution)
    kc.dynamic_prob2d(positions, values, pixels = pixels, max_workers = 1)
    kc.dynamic_prob2d(positions, values, xlim = [-10, 10], ylim = [-10, 10],
                      resolution = resolution)
    kc.dynamic_prob2d(positions, values, resolution = resolution)
    kc.dynamic_prob2d(positions, values, radii = 0, resolution = resolution)

    # Testing error cases
    with pytest.raises(ValueError):
        # Missing resolution
        kc.dynamic_prob2d(positions, values)

    with pytest.raises(AttributeError):
        # `pixels` has wrong type
        kc.dynamic_prob2d(positions, values, pixels = 0)

    with pytest.raises(ValueError):
        # `resolution` has wrong shape
        kc.dynamic_prob2d(positions, values, resolution = (1,))

    with pytest.raises(ValueError):
        # `resolution` has wrong numbers
        kc.dynamic_prob2d(positions, values, resolution = (1, 1))

    with pytest.raises(ValueError):
        # Wrong number of `values`
        kc.dynamic_prob2d(positions, np.ones(len(positions)),
                          resolution = (2, 2))


def test_static_prob2d():
    # Generate random 3D particles / trajectories
    np.random.seed(0)

    num_particles = 100
    resolution = (20, 20)

    positions = generate((num_particles, 2), -10, 10)
    values = generate(num_particles, 1, 2)
    radii = generate(num_particles, 0.1, 2)

    pixels = kc.static_prob2d(positions, values, radii = radii,
                              resolution = resolution)

    # Testing different settings work
    kc.static_prob2d(positions, values, radii = radii,
                     resolution = resolution)
    kc.static_prob2d(positions, values, pixels = pixels, max_workers = 1)
    kc.static_prob2d(positions, values, xlim = [-10, 10], ylim = [-10, 10],
                     resolution = resolution)
    kc.static_prob2d(positions, values, resolution = resolution)
    kc.static_prob2d(positions, values, radii = 0, resolution = resolution)

    # Testing error cases
    with pytest.raises(ValueError):
        # Missing resolution
        kc.static_prob2d(positions, values)

    with pytest.raises(AttributeError):
        # `pixels` has wrong type
        kc.static_prob2d(positions, values, pixels = 0)

    with pytest.raises(ValueError):
        # `resolution` has wrong shape
        kc.static_prob2d(positions, values, resolution = (1,))

    with pytest.raises(ValueError):
        # `resolution` has wrong numbers
        kc.static_prob2d(positions, values, resolution = (1, 1))

    with pytest.raises(ValueError):
        # Wrong number of `values`
        kc.static_prob2d(positions, np.ones(len(positions) - 1),
                         resolution = (2, 2))
