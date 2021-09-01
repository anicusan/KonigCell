#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : kc3d.pyx
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 30.08.2021


# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: embedsignature=True
# cython: cdivision=True


from libc.stdint cimport int32_t
cimport numpy as np


cdef extern from "../konigcell3d/konigcell3d.h":
    ctypedef struct kc3d_voxels:
        double  *grid
        int32_t *dims
        double  *xlim
        double  *ylim
        double  *zlim


    ctypedef struct kc3d_particles:
        double  *positions
        double  *radii
        double  *factors
        int32_t num_particles


    ctypedef enum kc3d_mode:
        kc3d_ratio = 0
        kc3d_intersection = 1
        kc3d_particle = 2
        kc3d_one = 3


    void kc3d_dynamic(
        kc3d_voxels             *voxels,
        const kc3d_particles    *particles,
        const kc3d_mode         mode,
        const int32_t           omit_last,
    ) nogil


    void kc3d_static(
        kc3d_voxels             *voxels,
        const kc3d_particles    *particles,
        const kc3d_mode         mode,
    ) nogil


cdef extern from "../konigcell3d/konigcell3d.c":
    # C is included here so that it doesn't need to be compiled externally
    pass


cpdef np.ndarray[double, ndim=3] dynamic3d(
    np.ndarray[double, ndim=3]  grid,
    const double[:]             xlim,
    const double[:]             ylim,
    const double[:]             zlim,
    const double[:, :]          positions,
    const int32_t               mode,
    const double[:]             radii = None,
    const double[:]             factors = None,
    bint                        omit_last = False,
):
    cdef int32_t        dims[3]
    cdef kc3d_voxels    voxels
    cdef kc3d_particles particles

    with nogil:
        dims[0] = grid.shape[0]
        dims[1] = grid.shape[1]
        dims[2] = grid.shape[1]

        voxels.grid = &grid[0, 0, 0]
        voxels.dims = &dims[0]
        voxels.xlim = &xlim[0]
        voxels.ylim = &ylim[0]
        voxels.zlim = &zlim[0]

        particles.positions = &positions[0, 0]
        particles.radii = &radii[0] if radii is not None else NULL
        particles.factors = &factors[0] if factors is not None else NULL
        particles.num_particles = positions.shape[0]

        if mode < 0 or mode > 3:
            raise ValueError((
                f"The input `mode` must be between 0 and 3! Received `{mode}`.\n"
                "Help:\n"
                "    kc.RATIO = 0          (= kc3d_ratio)\n"
                "    kc.INTERSECTION = 1   (= kc3d_intersection)\n"
                "    kc.PARTICLE = 2       (= kc3d_particle)\n"
                "    kc.ONE = 3            (= kc3d_one)\n"
            ))

        kc3d_dynamic(&voxels, &particles, <kc3d_mode>mode, <int32_t>omit_last)

    return grid


cpdef np.ndarray[double, ndim=3] static3d(
    np.ndarray[double, ndim=3]  grid,
    const double[:]             xlim,
    const double[:]             ylim,
    const double[:]             zlim,
    const double[:, :]          positions,
    const int32_t               mode,
    const double[:]             radii = None,
    const double[:]             factors = None,
):
    cdef int32_t        dims[3]
    cdef kc3d_voxels    voxels
    cdef kc3d_particles particles

    with nogil:
        dims[0] = grid.shape[0]
        dims[1] = grid.shape[1]
        dims[2] = grid.shape[2]

        voxels.grid = &grid[0, 0, 0]
        voxels.dims = &dims[0]
        voxels.xlim = &xlim[0]
        voxels.ylim = &ylim[0]
        voxels.zlim = &zlim[0]

        particles.positions = &positions[0, 0]
        particles.radii = &radii[0] if radii is not None else NULL
        particles.factors = &factors[0] if factors is not None else NULL
        particles.num_particles = positions.shape[0]

        if mode < 0 or mode > 3:
            raise ValueError((
                f"The input `mode` must be between 0 and 3! Received `{mode}`.\n"
                "Help:\n"
                "    kc.RATIO = 0          (= kc3d_ratio)\n"
                "    kc.INTERSECTION = 1   (= kc3d_intersection)\n"
                "    kc.PARTICLE = 2       (= kc3d_particle)\n"
                "    kc.ONE = 3            (= kc3d_one)\n"
            ))

        kc3d_static(&voxels, &particles, <kc3d_mode>mode)

    return grid
