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

import numpy    as      np
from tqdm       import  tqdm

from .voxels    import  Voxels
from .          import  kc3d, mode, utils




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
    '''Voxelize / rasterize a moving particle's trajectory onto a 3D voxel
    grid.

    This function is very general, so the description below may seem abstract;
    do check out the `examples` for specific applications and consult these
    docs if / as needed.

    The particle's `positions` are given as a 2D NumPy array, with three
    columns representing the X, Y and Z locations; each row corresponds to a
    single recorded particle position. Multiple trajectories can be separated
    by a row of NaN.

    As a particle moves in a straight line between two consecutive positions,
    it will be approximated as the convex hull of two spheres (centred at the
    given `positions`, with radii defined by the input `radii`) - like a
    cylinder with two spherical ends. Each such spherical cylinder defined by
    two consecutive particle positions will be rasterized onto a voxel grid.

    The values contained by the returned voxel grid depend on the input
    `mode`:

    1. `kc.RATIO`: each voxel will contain the ratio of its intersected volume
       to the total volume of the spherical cylinder defined by a particle
       moving between two adjacent positions. This is useful for e.g. velocity
       probability distributions.
    2. `kc.INTERSECTION`: each voxel will contain the intersected volume of the
       spherical cylinder; if the entire voxel was covered by the particle
       movement, it will simply contain the voxel volume `v_dx * v_dy * v_dz`.
       This is useful for e.g. residence time distributions.
    3. `kc.PARTICLE`: each voxel will contain the volume spanned by the
       spherical cylinder.
    4. `kc.ONE`: each intersected voxel will have one added to it.

    The volumes defined above can be multiplied by a particle-related quantity,
    e.g. particle velocity or time interval between two consecutive particle
    locations. They are defined by the input `values`; if unset all `values`
    are considered 1.

    The particles may have a radius; if the input `radii` is given as a single
    number, all particles will have this radius. Alternatively, the particle
    may change radius if `radii` is given as a NumPy vector containing a radius
    associated with each particle location. If unset (`None`), the particle
    is considered point-like / infinitesimally small.

    The voxel grid can be defined in two ways:

    1. Set the input `voxels` to a pre-existing `konigcell.Voxels` to reuse it,
       minimising memory allocation.
    2. Set the input `resolution` so that a new `konigcell.Voxels` will be
       created and returned by this function. If unset (`None`), `xlim`, `ylim`
       `zlim` will be computed automatically to contain all particle positions.

    The rasterization can be done in parallel, as defined by the input
    `executor`, which can be a ``concurrent.futures.Executor`` subclass; e.g.
    `ThreadPoolExecutor` for multi-threaded execution, or `mpi4py.futures`
    `MPIPoolExecutor` for distributing the computation across multiple MPI
    nodes. It can also be an *instance* of such class, configured beforehand.

    The number of parallel workers (threads, processes or ranks) is set by
    `max_workers`; if set to 1, execution is sequential (and produces better
    error messages). If unset, the `os.cpu_count()` is used.

    If `verbose` is True, the computation is timed and (hopefully) helpful
    messages are printed during execution.

    Parameters
    ----------
    positions: (N, 3) np.ndarray[ndim=2, dtype=float64]
        The 3D particle positions. Separate multiple trajectories with a row of
        NaN.

    mode: kc.RATIO, kc.INTERSECTION, kc.PARTICLE, kc.ONE
        The rasterization mode, see above for full details.

    values: float, (N-1,) np.ndarray, optional
        The particle values to rasterize, will be multiplied with what the mode
        returns; if a single `float`, all values are set to it. Multiple values
        can be given in a NumPy array for each particle movement, so it needs
        1 fewer values than positions (e.g. for particle positions A, B, C,
        there are only movements AB then BC). If unset (default), it is
        considered 1.

    radii: float, (N,) np.ndarray, optional
        Each particle's radius; if a single number, all particles are will have
        this radius. Multiple radii can be given in a NumPy array for each
        particle position. If `None` (default), the particle is considered
        point-like / infinitesimally small.

    voxels: konigcell.Voxels, optional
        A pre-created voxel grid to use; if unset, a new one will be created -
        so `resolution` must be set!

    resolution: 3-tuple, optional
        If `voxels` is unset, a new voxel grid will be created; this resolution
        contains the number of voxels in the X, Y and Z dimensions, e.g.
        ``(50, 50, 50)``. There must be at least 2 voxels in each dimension.

    xlim: 2-tuple, optional
        If `voxels` is unset, a new voxel grid will be created; you can
        manually set the physical rectangle spanned by this new grid as
        `[xmin, xmax]`. If unset, it is automatically computed to contain all
        particle positions.

    ylim: 2-tuple, optional
        Same as for `xlim`.

    zlim: 2-tuple, optional
        Same as for `xlim`.

    executor: concurrent.futures.Executor subclass or instance
        The parallel executor to use, implementing the `Executor` interface.
        For distributed computation with MPI, use `MPIPoolExecutor` from
        `mpi4py`. The default is `ThreadPoolExecutor`, which has the lowest
        overhead - useful because the main computation is done by C code
        releasing the GIL.

    max_workers: int, optional
        The maximum number of workers (threads, processes or ranks) to use by
        the parallel executor; if 1, it is sequential (and produces the
        clearest error messages should they happen). If unset, the
        `os.cpu_count()` is used.

    verbose: bool or str default True
        If `True`, time the computation and print the state of the execution.
        If `str`, show a message before loading bars.

    Examples
    --------
    TODO

    '''

    # Time voxellisation
    if verbose and not isinstance(verbose, str):
        start = time.time()

    # Type-checking inputs
    positions = np.asarray(positions, dtype = float, order = "C")

    if values is not None:
        if hasattr(values, "__iter__"):
            values = np.asarray(values, dtype = float, order = "C")
        else:
            values = float(values) * np.ones(len(positions) - 1, dtype = float)

    if radii is not None:
        if hasattr(radii, "__iter__"):
            radii = np.asarray(radii, dtype = float, order = "C")
        else:
            radii = float(radii) * np.ones(len(positions), dtype = float)

    utils.check_ncols(3, positions = positions)
    utils.check_lens(positions = positions, radii = radii)

    # Special case for dynamic3d: e.g. for 5 positions, there will be only 4
    # cylinders -> only need 4 values
    if values is not None and len(values) != len(positions[:-1]):
        raise ValueError(textwrap.fill((
            "For `dynamic3d`, there should be 1 fewer `values` than "
            f"`positions`. Received `len(positions) = {len(positions)}` and "
            f"`len(values) = {len(values)}`."
        )))

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

        shape = voxels.voxels.shape
        pos_chunks = utils.split(positions, max_workers, overlap = 1)
        rad_chunks = utils.split(radii, max_workers, overlap = 1)
        val_chunks = utils.split(values, max_workers, overlap = 1)
        vox_chunks = [np.zeros(shape) for _ in range(max_workers)]

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

    if verbose and not isinstance(verbose, str):
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
    '''Voxelize / rasterize static particles' positions onto a 3D voxel grid.

    This is exactly like the `dynamic3d` function, but particles are not
    considered to be moving - so they are rasterized a spheres.

    The input parameters are equivalent to `dynamic2d` - check its
    documentation for full details.

    Examples
    --------
    TODO

    '''

    # Time voxellisation
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

        shape = voxels.voxels.shape
        pos_chunks = utils.split(positions, max_workers)
        rad_chunks = utils.split(radii, max_workers)
        val_chunks = utils.split(values, max_workers)
        vox_chunks = [np.zeros(shape) for _ in range(max_workers)]

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

    if verbose and not isinstance(verbose, str):
        end = time.time()
        print(f"Voxellised in {end - start:3.3f} s")

    return voxels




def dynamic_prob3d(
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
    '''Compute the 3D probability distribution of a moving particle's specific
    quantity (e.g. velocity).

    This function computes the distribution of the input `values` across voxel
    cells. For example, computing the velocity distribution of a particle
    moving from `positions` A to B to C, we need to rasterize the velocity from
    `values[0]` on segment AB, then the velocity from `values[1]` on BC -
    therefore for N `positions` we will rasterize N-1 `values`.

    For multiple particle trajectories, simply separate them by a row of NaN
    in the input `positions`.

    All input parameters are equivalent to `dynamic3d` - check its
    documentation for full details.

    Examples
    --------
    TODO

    '''

    # Time voxellising
    if verbose:
        start = time.time()

    # Compute probability grid, where each voxel contains the values weighted
    # by the intersection volume; first compute values * weights...
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
    igrid.voxels[:, :, :] = 0.

    dynamic3d(
        positions,
        mode.INTERSECTION,
        radii = radii,
        voxels = igrid,
        executor = executor,
        max_workers = max_workers,
        verbose = "Step 2 / 2 :" if verbose else False,
    )

    nonzero = (igrid.voxels != 0.)

    if nonzero.any():
        voxels.voxels[nonzero] /= igrid.voxels[nonzero]

    # Correction for floating-point errors: threshold all voxels with values
    # below min(values); they can only exist due to FP errors
    minval = np.nanmin(values)
    voxels.voxels[voxels.voxels < minval] = minval

    if verbose:
        end = time.time()
        print(("Computed dynamic 3D probability distribution in "
               f"{end - start:3.3f} s"))

    return voxels




def static_prob3d(
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
    '''Compute the 3D probability distribution of static particles' specific
    quantities (e.g. velocity).

    This function computes the distribution of the input `values` across voxel
    cells for the static spherical particles at the input `positions`; it is
    the static counterpart of `dynamic_prob3d`.

    All input parameters are equivalent to `dynamic3d` - check its
    documentation for full details.

    Examples
    --------
    TODO

    '''

    # Time voxellising
    if verbose:
        start = time.time()

    # Compute probability grid, where each voxel contains the values weighted
    # by the intersection volume; first compute values * weights...
    voxels = static3d(
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
    igrid.voxels[:, :, :] = 0.

    static3d(
        positions,
        mode.INTERSECTION,
        radii = radii,
        voxels = igrid,
        executor = executor,
        max_workers = max_workers,
        verbose = "Step 2 / 2 :" if verbose else False,
    )

    nonzero = (igrid.voxels != 0.)

    if nonzero.any():
        voxels.voxels[nonzero] /= igrid.voxels[nonzero]

    # Correction for floating-point errors: threshold all voxels with values
    # below min(values); they can only exist due to FP errors
    minval = np.nanmin(values)
    voxels.voxels[voxels.voxels < minval] = minval

    if verbose:
        end = time.time()
        print(("Computed static 3D probability distribution in "
               f"{end - start:3.3f} s"))

    return voxels
