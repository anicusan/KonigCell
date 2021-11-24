/**
 * Quantitative, Fast Grid-Based Fields Calculations in 2D and 3D - Residence
 * Time Distributions, Velocity Grids, Eulerian Cell Projections etc.
 * 
 * If you used this codebase or any software making use of it in a scientific
 * publication, you are kindly asked to cite the following papers:
 * 
 *      <TODO: KonigCell Paper After Publication>
 *
 *      Powell D, Abel T. An exact general remeshing scheme applied to
 *      physically conservative voxelization. Journal of Computational
 *      Physics. 2015 Sep 15;297:340-56.
 *
 *
 * MIT License
 * 
 * Copyright (c) 2021 Andrei Leonard Nicusan
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


/**
 * The KonigCell2D core voxellisation routine is built on the R3D library (Powell and Abel, 2015
 * and LA-UR-15-26964). Andrei Leonard Nicusan modified the R3D code in 2021: rasterization was
 * optimised for `polyorder = 0` (i.e. only volume / zeroth moment); declarations and definitions
 * from the `r3d.h`, `v3d.h` and `r3d.c` were inlined; the power functions used were specialised
 * for single or double precision; variable-length arrays were removed for portability. For
 * consistency, the R3D prefix was changed to KC3D; *this does not remove attribution to the
 * original authors*.
 *
 * All rights for the R3D code go to the original authors of the library, whose copyright notice is
 * included below. A sincere thank you for your work.
 *
 * r3d.h
 * 
 * Routines for fast, geometrically robust clipping operations
 * and analytic volume/moment computations over polygons in 2D. 
 * 
 * Devon Powell
 * 31 August 2015
 *
 * This program was prepared by Los Alamos National Security, LLC at Los Alamos National
 * Laboratory (LANL) under contract No. DE-AC52-06NA25396 with the U.S. Department of Energy
 * (DOE). All rights in the program are reserved by the DOE and Los Alamos National Security,
 * LLC. Permission is granted to the public to copy and use this software without charge,
 * provided that this Notice and any statement of authorship are reproduced on all copies.
 * Neither the U.S. Government nor LANS makes any warranty, express or implied, or assumes any
 * liability or responsibility for the use of this software.
 */


#ifndef KONIGCELL3D_H
#define KONIGCELL3D_H


#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>


/**
 * User-changeable macros:
 *
 * KC3D_MAX_VERTS : maximum number of vertices used in the R3D internal polygon representation.
 * KC3D_SINGLE_PRECISION : if defined, single-precision floats will be used for calculations.
 */
#define KC3D_MAX_VERTS 1300
// #define KC3D_SINGLE_PRECISION


/**
 * Heap memory management functions. Define them before including KC3D for custom allocators.
 */
#ifndef KC3D_MALLOC
    #define KC3D_MALLOC malloc
#endif


#ifndef KC3D_CALLOC
    #define KC3D_CALLOC calloc
#endif


#ifndef KC3D_FREE
    #define KC3D_FREE free
#endif


/**
 * Use `restrict` pointers under C compilers.
 */
#ifdef __cplusplus
    #define KC3D_RESTRICT
#else
    #define KC3D_RESTRICT restrict
#endif


/**
 * Some types used by KonigCell3D - especially the geometry-related ones - are borrowed from the
 * R3D library (Powell and Abel, 2015 and LA-UR-15-26964); the `r3d_` prefixes were changed to
 * `kc3d_` for consistency. The following declarations are related to R3D. See below them for the
 * KonigCell3D functions.
 */


/* Real type used in calculations. */
#ifdef KC3D_SINGLE_PRECISION
typedef float kc3d_real;
#else 
typedef double kc3d_real;
#endif


/* Integer types used for indexing. */
typedef int32_t kc3d_int;
typedef int64_t kc3d_long;


/* A 3-vector. */
typedef union {
    struct
    {
        kc3d_real x,    /* x-component. */
                  y,    /* y-component. */
                  z;    /* z-component. */
    };
    kc3d_real xyz[3];   /* Index-based access to components. */
} kc3d_rvec3;


/* An integer 3-vector for grid indexing. */
typedef union {
    struct
    {
        kc3d_int i,     /* x-component. */
                 j,     /* y-component. */
                 k;     /* z-component. */
    };
    kc3d_int ijk[3];    /* Index-based access to components. */
} kc3d_dvec3;


/* A plane */
typedef struct {
	kc3d_rvec3 n;       /* Unit-length normal vector. */
	kc3d_real d;        /* Signed perpendicular distance to the origin. */
} kc3d_plane;


/* A doubly-linked vertex. */
typedef struct {
	kc3d_int pnbrs[3];  /* Neighbor indices. */
	kc3d_rvec3 pos;     /* Vertex position. */
} kc3d_vertex;


/* A polyhedron. Can be convex, nonconvex, even multiply-connected. */
typedef struct {
	kc3d_vertex verts[KC3D_MAX_VERTS];  /* Vertex buffer. */
	kc3d_int nverts;                    /* Number of vertices in the buffer. */
} kc3d_poly;


/* Like kc3d_init_poly, but assuming every face has exactly 3 indices */
/* Initialise a `kc3d_poly` from a vector of vertices and a vector of triangle face indices */
int kc3d_init_poly_tri(kc3d_poly *poly, kc3d_rvec3 *vertices, kc3d_int numverts,
                                        kc3d_dvec3 *faceinds, kc3d_int numfaces);





/**
 * KonigCell3D types and functions.
 */


/**
 * A voxel grid onto which `kc3d_dynamic` / `kc3d_static` / `kc3d_rasterize` voxellise a
 * trajectory / list of particle locations.
 *
 * Members
 * -------
 * grid : (D*H*W,) array
 *     A flattened C-ordered 3D array of voxels; has `dims[0] * dims[1] * dims[2]` elements. The
 *     voxel [i, j] is at index [i * dims[1] * dims[2] + j * dims[1] + k]. Must have at least 2x2x2
 *     voxels; non-nullable.
 *
 * dims : (3,) array
 *     The number of elements in the depth (x-dimension), height (y-dimension) and width
 *     (z-dimension) of the voxel grid.
 *
 * xlim : (2,) array
 *     The physical range spanned by `grid` in the x-dimension, formatted as [xmin, xmax].
 *
 * ylim : (2,) array
 *     The physical range spanned by `grid` in the y-dimension, formatted as [ymin, ymax].
 *
 * zlim : (2,) array
 *     The physical range spanned by `grid` in the z-dimension, formatted as [zmin, zmax].
 */
typedef struct {
    kc3d_real   *grid;
    kc3d_int    *dims;
    kc3d_real   *xlim;
    kc3d_real   *ylim;
    kc3d_real   *zlim;
} kc3d_voxels;


/**
 * Particle trajectory to be voxellised onto `kc3d_voxels` by `kc3d_dynamic` or `kc3d_static`.
 * Each trajectory segment will be multiplied by a `factor` - e.g. velocity for a velocity grid,
 * duration for a residence time distribution, 1 for a simple occupancy.
 *
 * A particle's size can change as it moves through a system (e.g. it diffuses / reacts) by varying
 * the `radii` values.
 *
 * Members
 * -------
 * positions : (3*num_particles,) array
 *     Particle 3D locations, in chronological order, formatted as [x0, y0, z0, x1, y1, z1, ...];
 *     has length `3 * num_particles`. Must have at least two locations; non-nullable.
 *
 * radii : (num_particles,) array or NULL
 *     The radius of each particle, formatted as [r0, r1, ...]; has length `num_particles`;
 *     if NULL, the particle trajectories are taken as extremely thin lines.
 *
 * factors : (num_particles,) array or NULL
 *     Factors to multiply each trajectory with, formatted as [f0, f1, ...]; e.g. the trajectory
 *     from [x0, y0, z0] to [x1, y1, z1] is multiplied by f0, from [x1, y1, z1] to [x2, y2, z2] is
 *     multiplied by f1; has length `num_particles - 1` for `kc3d_dynamic` and `num_particles` for
 *     `kc3d_static`. If NULL, all factors are taken as 1.
 *
 * num_particles
 *     The number of particles stored in the struct; see each member's definition above.
 */
typedef struct {
    kc3d_real   *positions;
    kc3d_real   *radii;
    kc3d_real   *factors;
    kc3d_int    num_particles;
} kc3d_particles;


/**
 * Voxellisation mode: every value added to the voxel grid will be multiplied by a volume-related
 * factor.
 *
 * For a residence time distribution the duration between two locations must be split across the
 * intersected voxels according to the ratio of each voxel's volume and the total trajectory
 * segment's voxel - that is `kc3d_ratio`.
 *
 * For a velocity grid, each particle's velocity must be added onto all intersected voxels,
 * regardless of the intersection volume - that is `kc3d_one`.
 *
 * Members
 * -------
 * kc3d_ratio
 *     Each value added to a voxel will be multiplied with the ratio of the particle-voxel
 *     intersection and the particle volume.
 *
 * kc3d_intersection
 *     Each value added to a voxel will be multiplied with the particle-voxel intersection.
 *
 * kc3d_particle
 *     Each value added to a voxel will be multiplied with the particle volume.
 *
 * kc3d_one
 *     Each value will be added to a voxel as-is, with no relation to the particle / voxel
 *     intersection volume.
 */
typedef enum {
    kc3d_ratio = 0,
    kc3d_intersection = 1,
    kc3d_particle = 2,
    kc3d_one = 3,
} kc3d_mode;


void            kc3d_dynamic(kc3d_voxels            *voxels,
                             const kc3d_particles   *particles,
                             const kc3d_mode        mode,
                             const kc3d_int         omit_last);


void            kc3d_static(kc3d_voxels             *voxels,
                            const kc3d_particles    *particles,
                            const kc3d_mode         mode);


void            kc3d_rasterize(kc3d_poly            *poly,
                               const kc3d_real      volume,
                               const kc3d_real      factor,
                               kc3d_voxels          *voxels,
                               kc3d_real            *local_grid,
                               const kc3d_mode      mode);


/**
 * Approximate a 3D sphere as a polygon with `KC3D_SC_NUM_VERTS` vertices. The input `poly` must be
 * pre-allocated; it will be initialised by this function. Returns analytical volume.
 */
kc3d_real       kc3d_sphere(kc3d_poly *poly,
                            const kc3d_rvec3 centre,
                            const kc3d_real radius);


/**
 * Approximate a 3D spherical cylinder (i.e. the convex hull of two oriented spherical halves)
 * between two points `p1` and `p2` with `KC3D_SC_NUM_VERTS` vertices *without the the second
 * spherical cap's volume*. The input `poly` must be pre-allocated; it will be initialised by this
 * function. Returns the analytical full spherical cylinder's volume.
 */
kc3d_real       kc3d_half_cylinder(kc3d_poly *poly,
                                   const kc3d_rvec3 p1,
                                   const kc3d_rvec3 p2,
                                   const kc3d_real r1,
                                   const kc3d_real r2);


/**
 * Approximate a 3D spherical cylinder (i.e. the convex hull of two oriented spherical halves)
 * between two points `p1` and `p2` with `KC3D_SC_NUM_VERTS` vertices. The input `poly` must be
 * pre-allocated; it will be initialised by this function. Returns the analytical full spherical
 * cylinder's volume.
 */
kc3d_real       kc3d_cylinder(kc3d_poly *poly,
                              const kc3d_rvec3 p1,
                              const kc3d_rvec3 p2,
                              const kc3d_real r1,
                              const kc3d_real r2);


#ifdef __cplusplus
}
#endif


#endif /*KONIGCELL3D_H*/
