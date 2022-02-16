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

import  numpy   as      np
from    tqdm    import  tqdm

from    .pixels import  Pixels
from    .       import  kc2d, mode, utils




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
    '''Pixelize / rasterize a moving particle's trajectory onto a 2D pixel
    grid.

    This function is very general, so the description below may seem abstract;
    do check out the `examples` for specific applications and consult these
    docs if / as needed.

    The particle's `positions` are given as a 2D NumPy array, with two columns
    representing the X and Y locations; each row corresponds to a single
    recorded particle position. Multiple trajectories can be separated by a row
    of NaN.

    As a particle moves in a straight line between two consecutive positions,
    it will be approximated as the convex hull of two circles (centred at the
    given `positions`, with radii defined by the input `radii`) - like a
    rectangle with two circular ends. Each such circular rectangle defined by
    two consecutive particle positions will be rasterized onto a pixel grid.

    The values contained by the returned pixel grid depend on the input
    `mode`:

    1. `kc.RATIO`: each pixel will contain the ratio of its intersected area to
       the total area of the circular rectangle defined by a particle moving
       between two adjacent positions. This is useful for e.g. velocity
       probability distributions.
    2. `kc.INTERSECTION`: each pixel will contain the intersected area of the
       circular rectangle; if the entire pixel was covered by the particle
       movement, it will simply contain the pixel area `p_dx * p_dy`. This is
       useful for e.g. residence time distributions.
    3. `kc.PARTICLE`: each pixel will contain the area spanned by the circular
       rectangle.
    4. `kc.ONE`: each intersected pixel will have one added to it.

    The areas defined above can be multiplied by a particle-related quantity,
    e.g. particle velocity or time interval between two consecutive particle
    locations. They are defined by the input `values`; if unset all `values`
    are considered 1.

    The particles may have a radius; if the input `radii` is given as a single
    number, all particles will have this radius. Alternatively, the particle
    may change radius if `radii` is given as a NumPy vector containing a radius
    associated with each particle location. If unset (`None`), the particle
    is considered point-like / infinitesimally small.

    The pixel grid can be defined in two ways:

    1. Set the input `pixels` to a pre-existing `konigcell.Pixels` to reuse it,
       minimising memory allocation.
    2. Set the input `resolution` so that a new `konigcell.Pixels` will be
       created and returned by this function. If unset (`None`), `xlim` and
       `ylim` will be computed automatically to contain all particle positions.

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
    positions : (N, 2) np.ndarray[ndim=2, dtype=float64]
        The 2D particle positions. Separate multiple trajectories with a row of
        NaN.

    mode : kc.RATIO, kc.INTERSECTION, kc.PARTICLE, kc.ONE
        The rasterization mode, see above for full details.

    values : float, (N-1,) np.ndarray, optional
        The particle values to rasterize, will be multiplied with what the mode
        returns; if a single `float`, all values are set to it. Multiple values
        can be given in a NumPy array for each particle movement, so it needs
        1 fewer values than positions (e.g. for particle positions A, B, C,
        there are only movements AB then BC). If unset (default), it is
        considered 1.

    radii : float, (N,) np.ndarray, optional
        Each particle's radius; if a single number, all particles are will have
        this radius. Multiple radii can be given in a NumPy array for each
        particle position. If `None` (default), the particle is considered
        point-like / infinitesimally small.

    pixels : kc.Pixels, optional
        A pre-created pixel grid to use; if unset, a new one will be created -
        so `resolution` must be set.

    resolution : 2-tuple, optional
        If `pixels` is unset, a new pixel grid will be created; this resolution
        contains the number of pixels in the X and Y dimensions, e.g.
        ``(512, 512)``. There must be at least 2 pixels in each dimension.

    xlim : 2-tuple, optional
        If `pixels` is unset, a new pixel grid will be created; you can
        manually set the physical rectangle spanned by this new grid as
        `[xmin, xmax]`. If unset, it is automatically computed to contain all
        particle positions.

    ylim : 2-tuple, optional
        Same as for `xlim`.

    executor : concurrent.futures.Executor subclass or instance
        The parallel executor to use, implementing the `Executor` interface.
        For distributed computation with MPI, use `MPIPoolExecutor` from
        `mpi4py`. The default is `ThreadPoolExecutor`, which has the lowest
        overhead - useful because the main computation is done by C code
        releasing the GIL.

    max_workers : int, optional
        The maximum number of workers (threads, processes or ranks) to use by
        the parallel executor; if 1, it is sequential (and produces the
        clearest error messages should they happen). If unset, the
        `os.cpu_count()` is used.

    verbose : bool or str default True
        If `True`, time the computation and print the state of the execution.
        If `str`, show a message before loading bars.

    Examples
    --------
    Compute the occupancy grid of a 2D particle of radius 0.5 moving between 3
    positions:

    >>> import numpy as np
    >>> positions = np.array([[0, 0], [1, 1], [2, 1]])
    >>>
    >>> import konigcell as kc
    >>> pixels = kc.dynamic2d(
    >>>     positions,
    >>>     kc.INTERSECTION,
    >>>     radii = 0.5,
    >>>     resolution = (500, 500),
    >>>     xlim = [-2, 5],
    >>>     ylim = [-2, 5],
    >>> )

    The ``kc.INTERSECTION`` pixellisation mode adds the area shaded by the
    particle's movement onto the grid.

    For a residence time distribution, compute the time difference between the
    particle's timestamps:

    >>> times = np.array([0, 2, 3])
    >>> dt = np.diff(times)
    >>> pixels = kc.dynamic2d(
    >>>     positions,
    >>>     kc.RATIO,
    >>>     values = dt,
    >>>     radii = 0.5,
    >>>     resolution = (500, 500),
    >>> )

    The ``kc.RATIO`` pixellisation mode splits a given value (in this
    case the time difference) across a particle's trajectory proportionally to
    the shaded area.

    If you omit ``xlim`` and ``ylim``, they will be computed automatically to
    include all particle positions.

    You can reuse the same pixels grid for multiple pixellisations:

    >>> kc.dynamic2d(
    >>>     positions,
    >>>     kc.INTERSECTION,
    >>>     radii = 0.5,
    >>>     pixels = pixels,
    >>> )

    In this case the ``resolution``, ``xlim`` and ``ylim`` don't need to be set
    anymore; they are extracted from the Pixels grid.

    You can set the maximum number of workers for the parallel pixellisation;
    e.g. for single-threaded execution:

    >>> pixels = kc.dynamic2d(
    >>>     positions,
    >>>     kc.ONE,
    >>>     radii = 0.5,
    >>>     resolution = (500, 500),
    >>>     max_workers = 1,
    >>> )

    The ``kc.ONE`` pixellisation flag simply adds 1 to each pixel intersected
    by the moving particle.

    Finally, here is a complete example computing the residence time
    distribution of a 2D particle moving randomly in a box of length 5:

    >>> positions = np.random.random((10_000, 2)) * 5
    >>> times = np.linspace(0, 100, 10_000)
    >>> dt = np.diff(times)
    >>>
    >>> rtd = kc.dynamic2d(
    >>>     positions,
    >>>     kc.RATIO,
    >>>     values = dt,
    >>>     radii = 0.5,
    >>>     resolution = (500, 500),
    >>> )
    '''
    # Time pixellisation
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

    utils.check_ncols(2, positions = positions)
    utils.check_lens(positions = positions, radii = radii)

    # Special case for dynamic2d: e.g. for 5 positions, there will be only 4
    # cylinders -> only need 4 values
    if values is not None and len(values) != len(positions[:-1]):
        raise ValueError(textwrap.fill((
            "For `dynamic2d`, there should be 1 fewer `values` than "
            f"`positions`. Received `len(positions) = {len(positions)}` and "
            f"`len(values) = {len(values)}`."
        )))

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
            offset = np.nanmax(radii) if radii is not None else 0
            xlim = utils.get_cutoffs(offset, positions[:, 0])

        if ylim is None:
            offset = np.nanmax(radii) if radii is not None else 0
            ylim = utils.get_cutoffs(offset, positions[:, 1])

        pixels = Pixels.zeros(resolution, xlim, ylim)

    # Pixellise according to execution policy
    if max_workers is not None and max_workers == 1:
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

        shape = pixels.pixels.shape
        pos_chunks = utils.split(positions, max_workers, overlap = 1)
        rad_chunks = utils.split(radii, max_workers, overlap = 1)
        val_chunks = utils.split(values, max_workers, overlap = 1)
        pix_chunks = [np.zeros(shape) for _ in range(max_workers)]

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
    '''Pixelize / rasterize static particles' positions onto a 2D pixel grid.

    This is exactly like the `dynamic2d` function, but particles are not
    considered to be moving - so they are rasterized a circles.

    The input parameters are equivalent to `dynamic2d` - check its
    documentation for full details.

    Examples
    --------
    Compute the occupancy grid of a 3 static 2D particles of radius 0.5:

    >>> import numpy as np
    >>> positions = np.array([[0, 0], [1, 1], [2, 1]])
    >>>
    >>> import konigcell as kc
    >>> pixels = kc.static2d(
    >>>     positions,
    >>>     kc.INTERSECTION,
    >>>     radii = 0.5,
    >>>     resolution = (500, 500),
    >>>     xlim = [-2, 5],
    >>>     ylim = [-2, 5],
    >>> )

    The ``kc.INTERSECTION`` pixellisation mode adds the area shaded by the
    particle's movement onto the grid.

    If you omit ``xlim`` and ``ylim``, they will be computed automatically to
    include all particle positions.

    You can reuse the same pixels grid for multiple pixellisations:

    >>> kc.static2d(
    >>>     positions,
    >>>     kc.INTERSECTION,
    >>>     radii = 0.5,
    >>>     pixels = pixels,
    >>> )

    In this case the ``resolution``, ``xlim`` and ``ylim`` don't need to be set
    anymore; they are extracted from the Pixels grid.

    You can set the maximum number of workers for the parallel pixellisation;
    e.g. for single-threaded execution:

    >>> pixels = kc.static2d(
    >>>     positions,
    >>>     kc.ONE,
    >>>     radii = 0.5,
    >>>     resolution = (500, 500),
    >>>     max_workers = 1,
    >>> )

    The ``kc.ONE`` pixellisation flag simply adds 1 to each pixel intersected
    by the moving particle.

    Finally, here is a complete example computing the occupancy grid of 10,000
    static 2D particles:

    >>> positions = np.random.random((10_000, 2)) * 5
    >>>
    >>> occupancy = kc.static2d(
    >>>     positions,
    >>>     kc.INTERSECTION,
    >>>     radii = 0.5,
    >>>     resolution = (500, 500),
    >>> )
    '''

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
            offset = np.nanmax(radii) if radii is not None else 0
            xlim = utils.get_cutoffs(offset, positions[:, 0])

        if ylim is None:
            offset = np.nanmax(radii) if radii is not None else 0
            ylim = utils.get_cutoffs(offset, positions[:, 1])

        pixels = Pixels.zeros(resolution, xlim, ylim)

    # Pixellise according to execution policy
    if max_workers is not None and max_workers == 1:
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

        shape = pixels.pixels.shape
        pos_chunks = utils.split(positions, max_workers)
        rad_chunks = utils.split(radii, max_workers)
        val_chunks = utils.split(values, max_workers)
        pix_chunks = [np.zeros(shape) for _ in range(max_workers)]

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




def dynamic_prob2d(
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
    '''Compute the 2D probability distribution of a moving particle's specific
    quantity (e.g. velocity).

    This function computes the distribution of the input `values` across pixel
    cells. For example, computing the velocity distribution of a particle
    moving from `positions` A to B to C, we need to rasterize the velocity from
    `values[0]` on segment AB, then the velocity from `values[1]` on BC -
    therefore for N `positions` we will rasterize N-1 `values`.

    For multiple particle trajectories, simply separate them by a row of NaN
    in the input `positions`.

    All input parameters are equivalent to `dynamic2d` - check its
    documentation for full details.
    '''

    # Time pixellising
    if verbose:
        start = time.time()

    # Compute probability grid, where each pixel contains the values weighted
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
    igrid.pixels[:, :] = 0.

    dynamic2d(
        positions,
        mode.INTERSECTION,
        radii = radii,
        pixels = igrid,
        executor = executor,
        max_workers = max_workers,
        verbose = "Step 2 / 2 :" if verbose else False,
    )

    nonzero = (igrid.pixels != 0.)

    if nonzero.any():
        pixels.pixels[nonzero] /= igrid.pixels[nonzero]

    # Correction for floating-point errors: threshold all pixels with values
    # below min(values); they can only exist due to FP errors
    minval = np.nanmin(values)
    pixels.pixels[pixels.pixels < minval] = minval

    if verbose:
        end = time.time()
        print(("Computed dynamic 2D probability distribution in "
               f"{end - start:3.3f} s"))

    return pixels




def static_prob2d(
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
    '''Compute the 2D probability distribution of static particles' specific
    quantities (e.g. velocity).

    This function computes the distribution of the input `values` across pixel
    cells for the static circular particles at the input `positions`; it is
    the static counterpart of `dynamic_prob2d`.

    All input parameters are equivalent to `dynamic2d` - check its
    documentation for full details.
    '''

    # Time pixellising
    if verbose:
        start = time.time()

    # Compute probability grid, where each pixel contains the values weighted
    # by the intersection area; first compute values * weights...
    pixels = static2d(
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
    igrid.pixels[:, :] = 0.

    static2d(
        positions,
        mode.INTERSECTION,
        radii = radii,
        pixels = igrid,
        executor = executor,
        max_workers = max_workers,
        verbose = "Step 2 / 2 :" if verbose else False,
    )

    nonzero = (igrid.pixels != 0.)

    if nonzero.any():
        pixels.pixels[nonzero] /= igrid.pixels[nonzero]

    # Correction for floating-point errors: threshold all pixels with values
    # below min(values); they can only exist due to FP errors
    minval = np.nanmin(values)
    pixels.pixels[pixels.pixels < minval] = minval

    if verbose:
        end = time.time()
        print(("Computed static 2D probability distribution in "
               f"{end - start:3.3f} s"))

    return pixels
