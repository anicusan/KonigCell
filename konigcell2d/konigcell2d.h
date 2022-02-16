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
 * The KonigCell2D core pixellisation routine is built on the R2D library (Powell and Abel, 2015
 * and LA-UR-15-26964). Andrei Leonard Nicusan modified the R2D code in 2021: rasterization was
 * optimised for `polyorder = 0` (i.e. only area / zeroth moment); declarations and definitions
 * from the `r2d.h`, `v2d.h` and `r2d.c` were inlined; the power functions used were specialised
 * for single or double precision; variable-length arrays were removed for portability. For
 * consistency, the R2D prefix was changed to KC2D; *this does not remove attribution to the
 * original authors*.
 *
 * All rights for the R2D code go to the original authors of the library, whose copyright notice is
 * included below. A sincere thank you for your work.
 *
 * r2d.h
 * 
 * Routines for fast, geometrically robust clipping operations
 * and analytic area/moment computations over polygons in 2D. 
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


#ifndef KONIGCELL2D_H
#define KONIGCELL2D_H


#ifdef __cplusplus
extern "C" {
#endif


#include <stdint.h>


/**
 * User-changeable macros:
 *
 * KC2D_NUM_VERTS : number of vertices used to approximate circles / cylinders as polygons.
 * KC2D_MAX_VERTS : maximum number of vertices used in the R2D internal polygon representation.
 * KC2D_SINGLE_PRECISION : if defined, single-precision floats will be used for calculations.
 */
#define KC2D_NUM_VERTS 32
#define KC2D_MAX_VERTS 50
// #define KC2D_SINGLE_PRECISION


/**
 * Heap memory management functions. Define them before including KC2D for custom allocators.
 */
#ifndef KC2D_MALLOC
    #define KC2D_MALLOC malloc
#endif


#ifndef KC2D_CALLOC
    #define KC2D_CALLOC calloc
#endif


#ifndef KC2D_FREE
    #define KC2D_FREE free
#endif


/**
 * Use `restrict` pointers under C compilers.
 */
#ifdef __cplusplus
    #define KC2D_RESTRICT
#else
    #define KC2D_RESTRICT restrict
#endif


/**
 * Some types used by KonigCell2D - especially the geometry-related ones - are borrowed from the
 * R2D library (Powell and Abel, 2015 and LA-UR-15-26964); the `r2d_` prefixes were changed to
 * `kc2d_` for consistency. The following declarations are related to R2D. See below them for the
 * KonigCell2D functions.
 */


/* Real type used in calculations. */
#ifdef KC2D_SINGLE_PRECISION
    typedef float kc2d_real;
#else 
    typedef double kc2d_real;
#endif


/* Integer types used for indexing. */
typedef int32_t kc2d_int;
typedef int64_t kc2d_long;


/* A 2-vector. */
typedef union {
    struct {
        kc2d_real x,    /* x-component. */
                  y;    /* y-component. */
    };
    kc2d_real xy[2];    /* Index-based access to components. */
} kc2d_rvec2;


/* An integer 2-vector for grid indexing. */
typedef union {
    struct {
        kc2d_int i,     /* x-component. */
                 j;     /* y-component. */
    };
    kc2d_int ij[2];     /* Index-based access to components. */
} kc2d_dvec2;


/* A plane */
typedef struct {
    kc2d_rvec2 n;       /* Unit-length normal vector. */
    kc2d_real d;        /* Signed perpendicular distance to the origin. */
} kc2d_plane;


/* A doubly-linked vertex. */
typedef struct {
    kc2d_int pnbrs[2];  /* Neighbor indices. */
    kc2d_rvec2 pos;     /* Vertex position. */
} kc2d_vertex;


/* A polygon. Can be convex, nonconvex, even multiply-connected. */
typedef struct {
    kc2d_vertex verts[KC2D_MAX_VERTS];   /* Vertex buffer. */
    kc2d_int nverts;                     /* Number of vertices in the buffer. */
} kc2d_poly;


/* Initialise a `kc2d_poly` from an array of vertices. */
void kc2d_init_poly(kc2d_poly* poly, kc2d_rvec2* vertices, kc2d_int numverts);




/**
 * KonigCell2D types and functions.
 */


/**
 * A pixel grid onto which `kc2d_dynamic` and `kc2d_static` pixellise a trajectory / list of
 * particle locations.
 *
 * Members
 * -------
 * grid : (N*M,) array
 *     A flattened C-ordered 2D array of pixels; has `dims[0] * dims[1]` elements. The pixel
 *     [i, j] is at index [i * dims[1] + j]. Must have at least 2x2 pixels; non-nullable.
 *
 * dims : (2,) array
 *     The number of rows / elements in the x-dimension (dims[0]) and the number of columns /
 *     elements in the y-dimension (dims[1]) in `grid`.
 *
 * xlim : (2,) array
 *     The physical range spanned by `grid` in the x-dimension, formatted as [xmin, xmax].
 *
 * ylim : (2,) array
 *     The physical range spanned by `grid` in the y-dimension, formatted as [ymin, ymax].
 */
typedef struct {
    kc2d_real   *grid;
    kc2d_int    *dims;
    kc2d_real   *xlim;
    kc2d_real   *ylim;
} kc2d_pixels;


/**
 * Particle trajectory to be pixellised onto `kc2d_pixels` by `kc2d_dynamic` or `kc2d_static`.
 * Each trajectory segment will be multiplied by a `factor` - e.g. velocity for a velocity grid,
 * duration for a residence time distribution, 1 for a simple occupancy.
 *
 * A particle's size can change as it moves through a system (e.g. it diffuses / reacts) via the
 * `radii` values.
 *
 * Members
 * -------
 * positions : (2*num_particles,) array
 *     Particle 2D locations, in chronological order, formatted as [x0, y0, x1, y1, ...]; has
 *     length `2 * num_particles`. Must have at least two locations; non-nullable.
 *
 * radii : (num_particles,) array or NULL
 *     The radius of each particle, formatted as [r0, r1, ...]; has length `num_particles`;
 *     if NULL, the particle trajectories are taken as extremely thin lines.
 *
 * factors : (num_particles,) array or NULL
 *     Factors to multiply each trajectory, formatted as [f0, f1, ...]; e.g. the trajectory from
 *     [x0, y0] to [x1, y1] is multiplied by f0, from [x1, y1] to [x2, y2] is multiplied by f1;
 *     has length `num_particles - 1` for `kc2d_dynamic` and `num_particles` for `kc2d_static`. If
 *     NULL, all factors are taken as 1.
 *
 * num_particles
 *     The number of particles stored in the struct; see each member's definition above.
 */
typedef struct {
    kc2d_real   *positions;
    kc2d_real   *radii;
    kc2d_real   *factors;
    kc2d_int    num_particles;
} kc2d_particles;


/**
 * Pixellisation mode: every value added to the pixel grid will be multiplied by an area-related
 * factor.
 *
 * For a residence time distribution the duration between two locations must be split across the
 * intersected pixels according to the ratio of each pixel's area and the total trajectory
 * segment's area - that is `kc2d_ratio`.
 *
 * For a velocity grid, each particle's velocity must be added onto all intersected pixels,
 * regardless of the intersection area - that is `kc2d_one`.
 *
 * Members
 * -------
 * kc2d_ratio
 *     Each value added to a pixel will be multiplied with the ratio of the particle-pixel
 *     intersection and the particle area.
 *
 * kc2d_intersection
 *     Each value added to a pixel will be multiplied with the particle-pixel intersection.
 *
 * kc2d_particle
 *     Each value added to a pixel will be multiplied with the whole particle area.
 *
 * kc2d_one
 *     Each value will be added to a pixel as-is, with no relation to the particle / pixel
 *     intersection area.
 */
typedef enum {
    kc2d_ratio = 0,
    kc2d_intersection = 1,
    kc2d_particle = 2,
    kc2d_one = 3,
} kc2d_mode;


void            kc2d_dynamic(kc2d_pixels            *pixels,
                             const kc2d_particles   *particles,
                             const kc2d_mode        mode,
                             const kc2d_int         omit_last);


void            kc2d_static(kc2d_pixels             *pixels,
                            const kc2d_particles    *particles,
                            const kc2d_mode         mode);


void            kc2d_rasterize(kc2d_poly            *poly,
                               const kc2d_real      area,
                               const kc2d_real      factor,
                               kc2d_pixels          *pixels,
                               kc2d_real            *local_grid,
                               const kc2d_mode      mode);


/**
 * Approximate a circle as a polygon with `KC2D_NUM_VERTS` vertices. The input `poly` must be
 * pre-allocated; it will be initialised by this function. Returns analytical area.
 */
kc2d_real       kc2d_circle(kc2d_poly *poly,
                            const kc2d_rvec2 centre,
                            const kc2d_real radius);


/**
 * Approximate a 2D cylinder (i.e. the convex hull of two circles) between two points `p1` and `p2`
 * with `KC2D_NUM_VERTS` vertices *without the the second circle's area*. The input `poly` must be
 * pre-allocated; it will be initialised by this function. Returns the analytical full cylinder's
 * area.
 */
kc2d_real       kc2d_half_cylinder(kc2d_poly *poly,
                                   const kc2d_rvec2 p1,
                                   const kc2d_rvec2 p2,
                                   const kc2d_real r1,
                                   const kc2d_real r2);


/**
 * Approximate a 2D cylinder (i.e. the convex hull of two circles) between two points `p1` and `p2`
 * with `KC2D_NUM_VERTS` vertices. The input `poly` must be pre-allocated; it will be initialised
 * by this function. Returns the analytical full cylinder's area.
 */
kc2d_real       kc2d_cylinder(kc2d_poly *poly,
                              const kc2d_rvec2 p1,
                              const kc2d_rvec2 p2,
                              const kc2d_real r1,
                              const kc2d_real r2);


#ifdef __cplusplus
}
#endif


#endif /*KONIGCELL2D_H*/
