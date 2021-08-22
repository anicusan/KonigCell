#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : kc2d.pyx
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.05.2021


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: cdivision=True


from libc.stdint cimport int32_t


cdef extern from "../konigcell2d/konigcell2d.h":
    ctypedef struct kc2d_pixels:
        double  *grid
        double  *igrid
        int32_t *dims
        double  *xlim
        double  *ylim


    ctypedef struct kc2d_particles:
        double  *positions
        double  *radii
        double  *factors
        int32_t num_particles


    ctypedef enum kc2d_mode:
        kc2d_ratio = 0
        kc2d_intersection = 1
        kc2d_particle = 2
        kc2d_one = 3


    void kc2d_dynamic(
        kc2d_pixels             *pixels,
        const kc2d_particles    *particles,
        const kc2d_mode         mode,
        const int32_t           omit_last,
    ) nogil


    void kc2d_static(
        kc2d_pixels             *pixels,
        const kc2d_particles    *particles,
        const kc2d_mode         mode,
    ) nogil


cdef extern from "../konigcell2d/konigcell2d.c":
    # C is included here so that it doesn't need to be compiled externally
    pass


cpdef void dynamic2d(
    double[:, :]        grid,
    const double[:]     xlim,
    const double[:]     ylim,
    const double[:, :]  positions,
    const int32_t       mode,
    const double[:]     radii = None,
    const double[:]     factors = None,
    double[:, :]        intersections = None,
    bint                omit_last = False,
) nogil except *:
    cdef int32_t        dims[2]
    cdef kc2d_pixels    pixels
    cdef kc2d_particles particles

    dims[0] = grid.shape[0]
    dims[1] = grid.shape[1]

    pixels.grid = &grid[0, 0]
    pixels.igrid = &intersections[0, 0] if intersections is not None else NULL
    pixels.dims = &dims[0]
    pixels.xlim = &xlim[0]
    pixels.ylim = &ylim[0]

    particles.positions = &positions[0, 0]
    particles.radii = &radii[0] if radii is not None else NULL
    particles.factors = &factors[0] if factors is not None else NULL
    particles.num_particles = positions.shape[0]

    if mode < 0 or mode > 3:
        raise ValueError((
            f"The input `mode` must be between 0 and 3! Received `{mode}`.\n"
            "Help:\n"
            "    kc.RATIO = 0          (= kc2d_ratio)\n"
            "    kc.INTERSECTION = 1   (= kc2d_intersection)\n"
            "    kc.PARTICLE = 2       (= kc2d_particle)\n"
            "    kc.ONE = 3            (= kc2d_one)\n"
        ))

    kc2d_dynamic(&pixels, &particles, <kc2d_mode>mode, <int32_t>omit_last)


cpdef void static2d(
    double[:, :]        grid,
    const double[:]     xlim,
    const double[:]     ylim,
    const double[:, :]  positions,
    const int32_t       mode,
    const double[:]     radii = None,
    const double[:]     factors = None,
    double[:, :]        intersections = None,
) nogil except *:
    cdef int32_t        dims[2]
    cdef kc2d_pixels    pixels
    cdef kc2d_particles particles

    dims[0] = grid.shape[0]
    dims[1] = grid.shape[1]

    pixels.grid = &grid[0, 0]
    pixels.igrid = &intersections[0, 0] if intersections is not None else NULL
    pixels.dims = &dims[0]
    pixels.xlim = &xlim[0]
    pixels.ylim = &ylim[0]

    particles.positions = &positions[0, 0]
    particles.radii = &radii[0] if radii is not None else NULL
    particles.factors = &factors[0] if factors is not None else NULL
    particles.num_particles = positions.shape[0]

    if mode < 0 or mode > 3:
        raise ValueError((
            f"The input `mode` must be between 0 and 3! Received `{mode}`.\n"
            "Help:\n"
            "    kc.RATIO = 0          (= kc2d_ratio)\n"
            "    kc.INTERSECTION = 1   (= kc2d_intersection)\n"
            "    kc.PARTICLE = 2       (= kc2d_particle)\n"
            "    kc.ONE = 3            (= kc2d_one)\n"
        ))

    kc2d_static(&pixels, &particles, <kc2d_mode>mode)
