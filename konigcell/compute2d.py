#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : compute2d.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 24.08.2021


import os
import time
import textwrap
from concurrent.futures import Executor, ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from .pixels import Pixels
from . import kc2d, mode, utils




def dynamic2d(
    positions,
    mode,
    values = None,
    radii = None,
    pixels = None,
    resolution = None,
    xlim = None,
    ylim = None,
    executor = ThreadPoolExecutor,
    max_workers = None,
    verbose = True,
):
    # Time pixellisation
    if verbose and not isinstance(verbose, str):
        start = time.time()

    # Type-checking inputs
    positions = np.asarray(positions, dtype = float, order = "C")

    if values is not None:
        if hasattr(values, "__iter__"):
            values = np.asarray(values, dtype = float, order = "C")
        else:
            values = float(values) * np.ones(len(positions), dtype = float)

    if radii is not None:
        if hasattr(radii, "__iter__"):
            radii = np.asarray(radii, dtype = float, order = "C")
        else:
            radii = float(radii) * np.ones(len(positions), dtype = float)

    utils.check_ncols(2, positions = positions)
    utils.check_lens(positions = positions, values = values, radii = radii)

    # If pixels is None, create a new Pixels grid
    if pixels is None:
        if resolution is None:
            raise ValueError("`resolution` must be defined if `pixels = None`")
        else:
            resolution = np.array(resolution, dtype = int)
            if resolution.ndim != 1 or len(resolution) != 2 or \
                    np.any(resolution < 2):
                raise ValueError(textwrap.fill((
                    "`resolution` must have exactly two elements (M, N), both "
                    f"larger than 2. Received `resolution = {resolution}`."
                )))

        if xlim is None:
            offset = radii.max() if radii is not None else 0
            xlim = utils.get_cutoffs(offset, positions[:, 0])

        if ylim is None:
            offset = radii.max() if radii is not None else 0
            ylim = utils.get_cutoffs(offset, positions[:, 1])

        pixels = Pixels.zeros(resolution, xlim, ylim)

    # Pixellise according to execution policy
    if executor == "seq" or (max_workers is not None and max_workers == 1):
        kc2d.dynamic2d(
            pixels.pixels,
            pixels.xlim,
            pixels.ylim,
            positions,
            mode,
            radii = radii,
            factors = values,
        )
    else:
        # Split `positions` into `max_workers` chunks for parallel processing
        if max_workers is None:
            max_workers = os.cpu_count()

        # Each chunk must have at least 2 positions
        if len(positions) / max_workers < 2:
            max_workers = len(positions) // 2

        pos_chunks = utils.split(positions, max_workers, overlap = 1)
        rad_chunks = utils.split(radii, max_workers, overlap = 1)
        val_chunks = utils.split(values, max_workers, overlap = 1)
        pix_chunks = [np.zeros(pixels.shape) for _ in range(max_workers)]

        # If `executor` is a class (rather than an instance / object),
        # instantiate it
        executor_isclass = False
        if not isinstance(executor, Executor):
            executor_isclass = True
            executor = executor(max_workers)

        # Pixellise each chunk separately
        futures = [
            executor.submit(
                kc2d.dynamic2d,
                pix_chunks[i],
                pixels.xlim,
                pixels.ylim,
                pos_chunks[i],
                mode,
                radii = rad_chunks[i],
                factors = val_chunks[i],
                omit_last = True,
            ) for i in range(max_workers - 1)
        ]

        # Pixellise the last chunk with omit_last = False
        futures.append(
            executor.submit(
                kc2d.dynamic2d,
                pix_chunks[-1],
                pixels.xlim,
                pixels.ylim,
                pos_chunks[-1],
                mode,
                radii = rad_chunks[-1],
                factors = val_chunks[-1],
                omit_last = False,
            )
        )

        # If verbose, show tqdm loading bar with text
        if verbose:
            desc = verbose if isinstance(verbose, str) else None
            futures = tqdm(futures, desc = desc)

        # Add all results onto the pixel grid
        pixels.pixels[:, :] += sum((f.result() for f in futures))

        # Clean up all copies immediately to free memory
        del futures, pos_chunks, rad_chunks, val_chunks, pix_chunks
        if executor_isclass:
            executor.shutdown()

    if verbose and not isinstance(verbose, str):
        end = time.time()
        print(f"Pixellised in {end - start:3.3f} s")

    return pixels




def static2d(
    positions,
    mode,
    values = None,
    radii = None,
    pixels = None,
    resolution = None,
    xlim = None,
    ylim = None,
    executor = ThreadPoolExecutor,
    max_workers = None,
    verbose = True,
):
    # Time pixellisation
    if verbose and not isinstance(verbose, str):
        start = time.time()

    # Type-checking inputs
    positions = np.asarray(positions, dtype = float, order = "C")

    if values is not None:
        if hasattr(values, "__iter__"):
            values = np.asarray(values, dtype = float, order = "C")
        else:
            values = float(values) * np.ones(len(positions), dtype = float)

    if radii is not None:
        if hasattr(radii, "__iter__"):
            radii = np.asarray(radii, dtype = float, order = "C")
        else:
            radii = float(radii) * np.ones(len(positions), dtype = float)

    utils.check_ncols(2, positions = positions)
    utils.check_lens(positions = positions, values = values, radii = radii)

    # If pixels is None, create a new Pixels grid
    if pixels is None:
        if resolution is None:
            raise ValueError("`resolution` must be defined if `pixels = None`")
        else:
            resolution = np.array(resolution, dtype = int)
            if resolution.ndim != 1 or len(resolution) != 2 or \
                    np.any(resolution < 2):
                raise ValueError(textwrap.fill((
                    "`resolution` must have exactly two elements (M, N), both "
                    f"larger than 2. Received `resolution = {resolution}`."
                )))

        if xlim is None:
            offset = radii.max() if radii is not None else 0
            xlim = utils.get_cutoffs(offset, positions[:, 0])

        if ylim is None:
            offset = radii.max() if radii is not None else 0
            ylim = utils.get_cutoffs(offset, positions[:, 1])

        pixels = Pixels.zeros(resolution, xlim, ylim)

    # Pixellise according to execution policy
    if executor == "seq" or (max_workers is not None and max_workers == 1):
        kc2d.static2d(
            pixels.pixels,
            pixels.xlim,
            pixels.ylim,
            positions,
            mode,
            radii = radii,
            factors = values,
        )
    else:
        # Split `positions` into `max_workers` chunks for parallel processing
        if max_workers is None:
            max_workers = os.cpu_count()

        # Each chunk must have at least 2 positions
        if len(positions) / max_workers < 2:
            max_workers = len(positions) // 2

        pos_chunks = utils.split(positions, max_workers)
        rad_chunks = utils.split(radii, max_workers)
        val_chunks = utils.split(values, max_workers)
        pix_chunks = [np.zeros(pixels.shape) for _ in range(max_workers)]

        # If `executor` is a class (rather than an instance / object),
        # instantiate it
        executor_isclass = False
        if not isinstance(executor, Executor):
            executor_isclass = True
            executor = executor(max_workers)

        # Pixellise each chunk separately
        futures = [
            executor.submit(
                kc2d.static2d,
                pix_chunks[i],
                pixels.xlim,
                pixels.ylim,
                pos_chunks[i],
                mode,
                radii = rad_chunks[i],
                factors = val_chunks[i],
            ) for i in range(max_workers)
        ]

        # If verbose, show tqdm loading bar with text
        if verbose:
            desc = verbose if isinstance(verbose, str) else None
            futures = tqdm(futures, desc = desc)

        # Add all results onto the pixel grid
        pixels.pixels[:, :] += sum((f.result() for f in futures))

        # Clean up all copies immediately to free memory
        del futures, pos_chunks, rad_chunks, val_chunks, pix_chunks
        if executor_isclass:
            executor.shutdown()

    if verbose and not isinstance(verbose, str):
        end = time.time()
        print(f"Pixellised in {end - start:3.3f} s")

    return pixels




def probability2d(
    positions,
    values,
    radii = None,
    pixels = None,
    resolution = None,
    xlim = None,
    ylim = None,
    executor = ThreadPoolExecutor,
    max_workers = None,
    verbose = True,
):
    # Time pixellising
    if verbose:
        start = time.time()

    # Compute velocity grid, where each pixel contains the values weighted
    # by the intersection area; first compute values * weights...
    pixels = dynamic2d(
        positions,
        mode.INTERSECTION,
        values = values,
        radii = radii,
        pixels = pixels,
        resolution = resolution,
        xlim = xlim,
        ylim = ylim,
        executor = executor,
        max_workers = max_workers,
        verbose = "Step 1 / 2 :" if verbose else False,
    )

    # ... then divide by the sum of weights
    igrid = pixels.copy()
    igrid[:, :] = 0.

    igrid = dynamic2d(
        positions,
        mode.INTERSECTION,
        radii = radii,
        pixels = igrid,
        executor = executor,
        max_workers = max_workers,
        verbose = "Step 2 / 2 :" if verbose else False,
    )

    pixels.pixels[igrid != 0.] /= igrid[igrid != 0.]

    if verbose:
        end = time.time()
        print(f"Computed 2D probability distribution in {end - start:3.3f} s")

    return pixels
