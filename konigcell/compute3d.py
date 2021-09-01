#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : compute3d.py
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 24.08.2021


import os
import time
import textwrap
from concurrent.futures import Executor, ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from .voxels import Voxels
from . import kc3d, mode, utils




def dynamic3d(
    positions,
    mode,
    values = None,
    radii = None,
    voxels = None,
    resolution = None,
    xlim = None,
    ylim = None,
    zlim = None,
    executor = ThreadPoolExecutor,
    max_workers = None,
    verbose = True,
):
    # Time voxellisation
    if not isinstance(verbose, str):
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

    utils.check_ncols(3, positions = positions)
    utils.check_lens(positions = positions, values = values, radii = radii)

    # If voxels is None, create a new Voxels grid
    if voxels is None:
        if resolution is None:
            raise ValueError("`resolution` must be defined if `voxels = None`")
        else:
            resolution = np.array(resolution, dtype = int)
            if resolution.ndim != 1 or len(resolution) != 3 or \
                    np.any(resolution < 2):
                raise ValueError(textwrap.fill((
                    "`resolution` must have exactly three elements (M, N, P), "
                    f"all larger than 3. Received `resolution = {resolution}`."
                )))

        offset = radii.max() if radii is not None else 0
        if xlim is None:
            xlim = utils.get_cutoffs(offset, positions[:, 0])

        if ylim is None:
            ylim = utils.get_cutoffs(offset, positions[:, 1])

        if zlim is None:
            zlim = utils.get_cutoffs(offset, positions[:, 2])

        voxels = Voxels.zeros(resolution, xlim, ylim, zlim)

    # Voxellise according to execution policy
    if executor == "seq" or (max_workers is not None and max_workers == 1):
        kc3d.dynamic3d(
            voxels.voxels,
            voxels.xlim,
            voxels.ylim,
            voxels.zlim,
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
        vox_chunks = [np.zeros(voxels.shape) for _ in range(max_workers)]

        # If `executor` is a class (rather than an instance / object),
        # instantiate it
        executor_isclass = False
        if not isinstance(executor, Executor):
            executor_isclass = True
            executor = executor(max_workers)

        # Voxellise each chunk separately
        futures = [
            executor.submit(
                kc3d.dynamic3d,
                vox_chunks[i],
                voxels.xlim,
                voxels.ylim,
                voxels.zlim,
                pos_chunks[i],
                mode,
                radii = rad_chunks[i],
                factors = val_chunks[i],
                omit_last = True,
            ) for i in range(max_workers - 1)
        ]

        # Voxellise the last chunk with omit_last = False
        futures.append(
            executor.submit(
                kc3d.dynamic3d,
                vox_chunks[-1],
                voxels.xlim,
                voxels.ylim,
                voxels.zlim,
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

        # Add all results onto the voxel grid
        voxels.voxels[:, :, :] += sum((f.result() for f in futures))

        # Clean up all copies immediately to free memory
        del futures, pos_chunks, rad_chunks, val_chunks, vox_chunks
        if executor_isclass:
            executor.shutdown()

    if not isinstance(verbose, str):
        end = time.time()
        print(f"Voxellised in {end - start:3.3f} s")

    return voxels




def static3d(
    positions,
    mode,
    values = None,
    radii = None,
    voxels = None,
    resolution = None,
    xlim = None,
    ylim = None,
    zlim = None,
    executor = ThreadPoolExecutor,
    max_workers = None,
    verbose = True,
):
    # Time voxellisation
    if not isinstance(verbose, str):
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

    utils.check_ncols(3, positions = positions)
    utils.check_lens(positions = positions, values = values, radii = radii)

    # If voxels is None, create a new Voxels grid
    if voxels is None:
        if resolution is None:
            raise ValueError("`resolution` must be defined if `voxels = None`")
        else:
            resolution = np.array(resolution, dtype = int)
            if resolution.ndim != 1 or len(resolution) != 3 or \
                    np.any(resolution < 2):
                raise ValueError(textwrap.fill((
                    "`resolution` must have exactly three elements (M, N, P), "
                    f"all larger than 2. Received `resolution = {resolution}`."
                )))

        offset = radii.max() if radii is not None else 0

        if xlim is None:
            xlim = utils.get_cutoffs(offset, positions[:, 0])

        if ylim is None:
            ylim = utils.get_cutoffs(offset, positions[:, 1])

        if zlim is None:
            zlim = utils.get_cutoffs(offset, positions[:, 2])

        voxels = Voxels.zeros(resolution, xlim, ylim, zlim)

    # Voxellise according to execution policy
    if executor == "seq" or (max_workers is not None and max_workers == 1):
        kc3d.static3d(
            voxels.voxels,
            voxels.xlim,
            voxels.ylim,
            voxels.zlim,
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
        vox_chunks = [np.zeros(voxels.shape) for _ in range(max_workers)]

        # If `executor` is a class (rather than an instance / object),
        # instantiate it
        executor_isclass = False
        if not isinstance(executor, Executor):
            executor_isclass = True
            executor = executor(max_workers)

        # Voxellise each chunk separately
        futures = [
            executor.submit(
                kc3d.static3d,
                vox_chunks[i],
                voxels.xlim,
                voxels.ylim,
                voxels.zlim,
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

        # Add all results onto the voxel grid
        voxels.voxels[:, :, :] += sum((f.result() for f in futures))

        # Clean up all copies immediately to free memory
        del futures, pos_chunks, rad_chunks, val_chunks, vox_chunks
        if executor_isclass:
            executor.shutdown()

    if verbose:
        end = time.time()
        print(f"Voxellised in {end - start:3.3f} s")

    return voxels




def probability3d(
    positions,
    values,
    radii = None,
    voxels = None,
    resolution = None,
    xlim = None,
    ylim = None,
    zlim = None,
    executor = ThreadPoolExecutor,
    max_workers = None,
    verbose = True,
):
    # Time voxellising
    if verbose:
        start = time.time()

    # Compute velocity grid, where each voxel contains the values weighted
    # by the intersection area; first compute values * weights...
    voxels = dynamic3d(
        positions,
        mode.INTERSECTION,
        values = values,
        radii = radii,
        voxels = voxels,
        resolution = resolution,
        xlim = xlim,
        ylim = ylim,
        zlim = zlim,
        executor = executor,
        max_workers = max_workers,
        verbose = "Step 1 / 2 :" if verbose else False,
    )

    # ... then divide by the sum of weights
    igrid = voxels.copy()
    igrid[:, :, :] = 0.

    igrid = dynamic3d(
        positions,
        mode.INTERSECTION,
        radii = radii,
        voxels = igrid,
        executor = executor,
        max_workers = max_workers,
        verbose = "Step 2 / 2 :" if verbose else False,
    )

    voxels.voxels[igrid != 0.] /= igrid[igrid != 0.]

    if verbose:
        end = time.time()
        print(f"Computed 3D probability distribution in {end - start:3.3f} s")

    return voxels
