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
 * for single or double precision; variable-length arrays were removed for portability.
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
 * Neither the U.S. 
 *
 * Government nor LANS makes any warranty, express or implied, or assumes any liability 
 * or responsibility for the use of this software.
 */


#include "konigcell2d.h"


#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>




/* Functions from R2D used by KonigCell2D. See https://github.com/devonmpowell/r3d */
void kc2d_translate(kc2d_poly* poly, kc2d_rvec2 shift);
void kc2d_get_ibox(kc2d_poly* poly, kc2d_dvec2 ibox[2], kc2d_rvec2 d);
void kc2d_clamp_ibox(kc2d_poly* poly, kc2d_dvec2 ibox[2], kc2d_dvec2 clampbox[2], kc2d_rvec2 d);
void kc2d_clip(kc2d_poly* poly, kc2d_plane* planes, kc2d_int nplanes);
void kc2d_reduce(kc2d_poly* poly, kc2d_real* moments);
void kc2d_split_coord(kc2d_poly* inpoly, kc2d_poly** outpolys, kc2d_real coord, kc2d_int ax);
void kc2d_rasterize_local(kc2d_poly* poly, kc2d_dvec2 ibox[2], kc2d_real* dest_grid, kc2d_rvec2 d);




#define KC2D_PI 3.14159265358979323846


#ifdef SINGLE_PRECISION
    #define KC2D_SQRT(x) sqrtf(x)
    #define KC2D_COS(x) cosf(x)
    #define KC2D_SIN(x) sinf(x)
    #define KC2D_ACOS(x) acosf(x)
    #define KC2D_ATAN2(x, y) atan2f((x), (y))
    #define KC2D_FLOOR(x) floorf(x)
#else
    #define KC2D_SQRT(x) sqrt(x)
    #define KC2D_COS(x) cos(x)
    #define KC2D_SIN(x) sin(x)
    #define KC2D_ACOS(x) acos(x)
    #define KC2D_ATAN2(x, y) atan2((x), (y))
    #define KC2D_FLOOR(x) floor(x)
#endif




/* Absolute value */
kc2d_real       kc2d_fabs(kc2d_real x)
{
    return (x >= 0 ? x : -x);
}


/* Euclidean distance between two points defined by (x1, y1) and (x2, y2) */
kc2d_real       kc2d_dist(kc2d_real x1, kc2d_real y1, kc2d_real x2, kc2d_real y2)
{
    return KC2D_SQRT((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}


/* Squared Euclidean distance between two points defined by (x1, y1) and (x2, y2) */
kc2d_real       kc2d_dist2(kc2d_real x1, kc2d_real y1, kc2d_real x2, kc2d_real y2)
{
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}


/* Square a number */
kc2d_real       kc2d_pow2(kc2d_real x)
{
    return x * x;
}


/* Area of a triangle defined by three points (x1, y1), (x2, y2), (x3, y3) */
kc2d_real       kc2d_triangle(kc2d_real x1, kc2d_real y1,
                              kc2d_real x2, kc2d_real y2,
                              kc2d_real x3, kc2d_real y3)
{
    return 0.5 * kc2d_fabs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2));
}


/* Distance from a point (x0, y0) to a line defined by two points (x1, y1) and (x2, y2) */
kc2d_real       kc2d_dist_pl(kc2d_real x0, kc2d_real y0,
                             kc2d_real x1, kc2d_real y1,
                             kc2d_real x2, kc2d_real y2)
{
    // The vectorial form here is much more numerically stable than the classic division formula
    // Line defined by a position vector `a` and unit-length direction vector `n`
    kc2d_real   ax = x1;
    kc2d_real   ay = y1;

    // Auxilliaries
    kc2d_real   d = kc2d_dist(x1, y1, x2, y2);
    kc2d_real   ax0 = x0 - ax;
    kc2d_real   ay0 = y0 - ay;

    kc2d_real   nx = (x2 - x1) / d;
    kc2d_real   ny = (y2 - y1) / d;

    // Dot product between (p - a) and n
    kc2d_real   k = ax0 * nx + ay0 * ny;

    // Distance vector
    kc2d_real   dx = ax0 - k * nx;
    kc2d_real   dy = ay0 - k * ny;

    return KC2D_SQRT(dx * dx + dy * dy);
}


/* Area of a circular segment from a circle of radius r and a line at a distance d from the edge */
kc2d_real       kc2d_circular_segment(kc2d_real r, kc2d_real h)
{
    if (h < 0. || h > 2 * r)    // Possible in case of precision errors
        return 0.0;

    kc2d_real a = r * r * KC2D_ACOS((r - h) / r) - (r - h) * KC2D_SQRT(2 * r * h - h * h);
    // Truncate circular segment area in case of numerical precision errors
    if (a < 0.)
        return 0.;
    else if (a > KC2D_PI * r * r)
        return KC2D_PI * r * r;
    return a;
}


/**
 * Approximate a circle as a polygon with `KC2D_NUM_VERTS` vertices. The input `poly` must be
 * pre-allocated; it will be initialised by this function. Returns analytical area.
 */
kc2d_real       kc2d_circle(kc2d_poly *poly,
                            const kc2d_rvec2 centre,
                            const kc2d_real radius)
{
    kc2d_real    inc;
    kc2d_int     i;

    // Initialise neighbour indices; vertices will be set directly
    kc2d_init_poly(poly, NULL, KC2D_NUM_VERTS);
    kc2d_vertex *verts = poly->verts;

    inc = 2 * KC2D_PI / KC2D_NUM_VERTS;
    for (i = 0; i < KC2D_NUM_VERTS; ++i)
    {
        verts[i].pos.x = centre.x + KC2D_COS(i * inc) * radius;
        verts[i].pos.y = centre.y + KC2D_SIN(i * inc) * radius;
    }

    return KC2D_PI * radius * radius;
}


/**
 * Approximate a 2D cylinder (i.e. the convex hull of two circles) between two points `p1` and `p2`
 * with `KC2D_NUM_VERTS` vertices *without the the second circle's area*. The input `poly` must be
 * pre-allocated; it will be initialised by this function. Returns the analytical area.
 */
kc2d_real       kc2d_half_cylinder(kc2d_poly *poly,
                                   const kc2d_rvec2 p1,
                                   const kc2d_rvec2 p2,
                                   const kc2d_real r1,
                                   const kc2d_real r2)
{
    kc2d_real    ang;
    kc2d_real    start_ang, end_ang;
    kc2d_real    inc;

    kc2d_int     NUM_VERTS_2 = KC2D_NUM_VERTS / 2;
    kc2d_int     i;

    // Initialise neighbour indices; vertices will be set directly
    kc2d_init_poly(poly, NULL, KC2D_NUM_VERTS);
    kc2d_vertex *verts = poly->verts;

    // Get the angle between [0, 2pi] using an atan2 trick (atan2 returns [-pi, pi])
    ang = KC2D_PI - KC2D_ATAN2(p2.y - p1.y, -(p2.x - p1.x));

    start_ang = KC2D_PI / 2 + ang;
    end_ang = 3 * KC2D_PI / 2 + ang;

    inc = (end_ang - start_ang) / (NUM_VERTS_2 - 1);
    for (i = 0; i < NUM_VERTS_2; ++i)
    {
        verts[i].pos.x = p1.x + KC2D_COS(start_ang + i * inc) * r1;
        verts[i].pos.y = p1.y + KC2D_SIN(start_ang + i * inc) * r1;
    }

    inc = (start_ang - end_ang) / (NUM_VERTS_2 - 1);
    for (i = 0; i < NUM_VERTS_2; ++i)
    {
        verts[i + NUM_VERTS_2].pos.x = p2.x + KC2D_COS(end_ang + i * inc) * r2;
        verts[i + NUM_VERTS_2].pos.y = p2.y + KC2D_SIN(end_ang + i * inc) * r2;
    }

    return kc2d_dist(p1.x, p1.y, p2.x, p2.y) * (r1 + r2) + KC2D_PI / 2 * (r1 * r1 - r2 * r2);
}


/**
 * Approximate a 2D cylinder (i.e. the convex hull of two circles) between two points `p1` and `p2`
 * with `KC2D_NUM_VERTS` vertices. The input `poly` must be pre-allocated; it will be initialised
 * by this function. Returns the analytical full cylinder's area.
 */
kc2d_real       kc2d_cylinder(kc2d_poly *poly,
                              const kc2d_rvec2 p1,
                              const kc2d_rvec2 p2,
                              const kc2d_real r1,
                              const kc2d_real r2)
{
    kc2d_real    ang;
    kc2d_real    start_ang, end_ang;
    kc2d_real    inc;

    kc2d_int     NUM_VERTS_2 = KC2D_NUM_VERTS / 2;
    kc2d_int     i;

    // Initialise neighbour indices; vertices will be set directly
    kc2d_init_poly(poly, NULL, KC2D_NUM_VERTS);
    kc2d_vertex *verts = poly->verts;

    // Get the angle between [0, 2pi] using an atan2 trick (atan2 returns [-pi, pi])
    ang = KC2D_PI - KC2D_ATAN2(p2.y - p1.y, -(p2.x - p1.x));

    start_ang = KC2D_PI / 2 + ang;
    end_ang = 3 * KC2D_PI / 2 + ang;

    inc = (end_ang - start_ang) / (NUM_VERTS_2 - 1);
    for (i = 0; i < NUM_VERTS_2; ++i)
    {
        verts[i].pos.x = p1.x + KC2D_COS(start_ang + i * inc) * r1;
        verts[i].pos.y = p1.y + KC2D_SIN(start_ang + i * inc) * r1;
    }

    start_ang = ang - KC2D_PI / 2;
    end_ang = ang + KC2D_PI / 2;

    inc = (end_ang - start_ang) / (NUM_VERTS_2 - 1);
    for (i = 0; i < NUM_VERTS_2; ++i)
    {
        verts[i + NUM_VERTS_2].pos.x = p2.x + KC2D_COS(start_ang + i * inc) * r2;
        verts[i + NUM_VERTS_2].pos.y = p2.y + KC2D_SIN(start_ang + i * inc) * r2;
    }

    return kc2d_dist(p1.x, p1.y, p2.x, p2.y) * (r1 + r2) + KC2D_PI / 2 * (r1 * r1 + r2 * r2);
}


/**
 * Rasterize a polygon `poly` with area `area` onto a pixel grid `grid` with `dims` rows and
 * columns and xy `grid_size`, using a local `lgrid` for temporary calculations in the pixels
 * spanning the rectangular approximation of the polygon.
 *
 * The area ratio is multiplied by `factor` and *added* onto the global `grid`.
 * The local grid `lgrid` is reinitialised to zero at the end of the function.
 */
void            kc2d_rasterize_ll(kc2d_poly* KC2D_RESTRICT  poly,
                                  kc2d_real                 area,
                                  kc2d_real* KC2D_RESTRICT  grid,
                                  kc2d_real* KC2D_RESTRICT  lgrid,
                                  const kc2d_int            dims[2],
                                  const kc2d_rvec2          grid_size,
                                  const kc2d_real           factor,
                                  const kc2d_mode           mode)
{
    kc2d_dvec2   clampbox[2] = {{{0, 0}}, {{dims[0], dims[1]}}};
    kc2d_dvec2   ibox[2];        // Local grid's range of indices in the global grid
    kc2d_int     lx, ly;         // Local grid's written number of rows and columns
    kc2d_int     i, j;           // Iterators

    // Find the range of indices spanned by `poly`, then clamp them if `poly` extends out of `grid`
    kc2d_get_ibox(poly, ibox, grid_size);
    kc2d_clamp_ibox(poly, ibox, clampbox, grid_size);

    // Initialise local grid for the pixellisation step
    lx = ibox[1].i - ibox[0].i;
    ly = ibox[1].j - ibox[0].j;

    // Rasterize the polygon onto the local grid
    kc2d_rasterize_local(poly, ibox, lgrid, grid_size);

#define KC2D_GIDX (i * dims[1] + j)                             /* Global grid index */
#define KC2D_LIDX ((i - ibox[0].i) * ly + (j - ibox[0].j))      /* Local  grid index */

    // Add values from the local grid to the global one, depending on the pixellisation `mode`
    if (mode == kc2d_ratio)
    {
        for (i = ibox[0].i; i < ibox[1].i; ++i)
            for (j = ibox[0].j; j < ibox[1].j; ++j)
                grid[KC2D_GIDX] += factor * lgrid[KC2D_LIDX] / area;
    }

    else if (mode == kc2d_intersection)
    {
        for (i = ibox[0].i; i < ibox[1].i; ++i)
            for (j = ibox[0].j; j < ibox[1].j; ++j)
                grid[KC2D_GIDX] += factor * lgrid[KC2D_LIDX];
    }

    else if (mode == kc2d_particle)
    {
        for (i = ibox[0].i; i < ibox[1].i; ++i)
            for (j = ibox[0].j; j < ibox[1].j; ++j)
                if (lgrid[(i - ibox[0].i) * ly + j - ibox[0].j] != 0.)
                    grid[KC2D_GIDX] += factor * area;
    }

    else if (mode == kc2d_one)
    {
        for (i = ibox[0].i; i < ibox[1].i; ++i)
            for (j = ibox[0].j; j < ibox[1].j; ++j)
                if (lgrid[(i - ibox[0].i) * ly + j - ibox[0].j] != 0.)
                    grid[KC2D_GIDX] += factor;
    }

#undef KC2D_GIDX
#undef KC2D_LIDX

    // Reinitialise the written local grid to zero
    for (i = 0; i < lx * ly; ++i)
        lgrid[i] = 0.;
}


/* Check if the particle position at index `ip` is valid. */
kc2d_int        kc2d_valid_position(const kc2d_particles *particles, const kc2d_int ip)
{
    // Extract attributes needed
    kc2d_real   x = particles->positions[2 * ip];
    kc2d_real   y = particles->positions[2 * ip + 1];
    kc2d_real   r = (particles->radii == NULL ? 1e-6 : particles->radii[ip]);
    kc2d_real   f = (particles->factors == NULL ? 1. : particles->factors[ip]);

    return !(isnan(x) || isnan(y) || isnan(r) || isnan(f));
}


/* Find next valid trajectory's starting index, after `start`. */
kc2d_int        kc2d_next_segment_start(const kc2d_particles    *particles,
                                        const kc2d_int          start)
{
    kc2d_int    ip = start;

    while (ip < particles->num_particles)
    {
        // Segments must have at least two valid positions
        if (ip < particles->num_particles - 1 && kc2d_valid_position(particles, ip) &&
                kc2d_valid_position(particles, ip + 1))
            break;
        ++ip;
    }

    return ip;
}

/**
 * Find the start (inclusive) and end (exclusive) indices of the next trajectory segment after
 * `start`; save indices in `segment_bounds`. A trajectory segment is separated by NaNs. If no
 * valid segment exists, return 0; otherwise return the starting index of the *next* segment.
 */
kc2d_int        kc2d_next_segment(kc2d_int              *segment_bounds,
                                  const kc2d_particles  *particles,
                                  const kc2d_int        start)
{
    kc2d_int    ip;

    // Find starting index (inclusive)
    ip = kc2d_next_segment_start(particles, start);
    if (ip >= particles->num_particles)
        return 0;

    segment_bounds[0] = ip;

    // Find ending index (exclusive)
    ip = segment_bounds[0] + 2;         // Already checked next one is valid too
    while (ip < particles->num_particles && kc2d_valid_position(particles, ip))
        ++ip;

    segment_bounds[1] = ip;

    // Find next segment's start
    return kc2d_next_segment_start(particles, ip);
}


/**
 * Compute the occupancy grid of a single circular *moving* particle's trajectory.
 *
 * This corresponds to the pixellisation of moving circular particles, such that for every two
 * consecutive particle locations, a 2D cylinder (i.e. convex hull of two circles at the two
 * particle positions), the fraction of its area that intersets a pixel is multiplied with the
 * time between the two particle locations and saved in the input `pixels`.
 */
void            kc2d_dynamic(kc2d_pixels            *pixels,
                             const kc2d_particles   *particles,
                             const kc2d_mode        mode,
                             const kc2d_int         omit_last)
{
    // Some cheap input parameter checks
    if (pixels->dims[0] < 2 || pixels->dims[1] < 2 || particles->num_particles < 2)
    {
        fprintf(stderr, "[ERROR]: The input grid should have at least 2x2 cells, and there should "
                "be at least two particle positions.\n\n");
        return;
    }

    // Extract members from `pixels` and `particles`
    kc2d_real        *grid = pixels->grid;
    const kc2d_int   *dims = pixels->dims;
    const kc2d_real  *xlim = pixels->xlim;
    const kc2d_real  *ylim = pixels->ylim;

    const kc2d_real  *positions = particles->positions;
    const kc2d_real  *radii = particles->radii;
    const kc2d_real  *factors = particles->factors;
    const kc2d_int   num_particles = particles->num_particles;

    // Current trajectory segment bounds indices: start (inclusive), end (exclusive)
    kc2d_int        segment_bounds[2];
    kc2d_int        next = 0;

    // Auxilliaries
    kc2d_int        ip;             // Trajectory particle index
    kc2d_real       r1, r2;         // Radii for two particle
    kc2d_real       area;           // Total area for one 2D cylinder
    kc2d_real       factor;         // Current factor to multiply raster with

    // Initialise global pixel grid
    kc2d_real       xsize = xlim[1] - xlim[0];
    kc2d_real       ysize = ylim[1] - ylim[0];

    kc2d_rvec2      grid_size = {{xsize / dims[0], ysize / dims[1]}};
    kc2d_real       rsmall = 1.0e-6 * (grid_size.x < grid_size.y ? grid_size.x : grid_size.y);

    // Local grid which will be used for rasterising
    kc2d_real       *lgrid = (kc2d_real*)KC2D_CALLOC((size_t)dims[0] * dims[1], sizeof(kc2d_real));

    // Polygonal shapes used for the particle trajectories
    kc2d_poly       cylinder;

    // Copy `positions` to new local array and translate them such that the grid origin is (0, 0)
    kc2d_rvec2      *trajectory = (kc2d_rvec2*)KC2D_MALLOC(sizeof(kc2d_rvec2) * num_particles);

    for (ip = 0; ip < num_particles; ++ip)
    {
        trajectory[ip].x = positions[2 * ip] - xlim[0];
        trajectory[ip].y = positions[2 * ip + 1] - ylim[0];
    }

    // Rasterize particle trajectory segments: for each segment, across each two consecutive
    // particle positions, create a polygonal approximation of the convex hull of the two particle
    // locations, minus the second circle's area (which is added in the previous iteration)
    //
    // Find the next trajectory segment's index bounds and return the future one's start index
    while ((next = kc2d_next_segment(segment_bounds, particles, next)))
    {
        for (ip = segment_bounds[0]; ip < segment_bounds[1] - 1; ++ip)
        {
            r1 = (radii == NULL ? rsmall : radii[ip]);
            r2 = (radii == NULL ? rsmall : radii[ip + 1]);
            factor = (factors == NULL ? 1 : factors[ip]);

            // If this is the last cylinder from a segment, pixellise full cylinder, including
            // circular cap - unless it's the last segment and omit_last
            if (ip == segment_bounds[1] - 2 && !(next >= num_particles - 1 && omit_last))
            {
                area = kc2d_cylinder(&cylinder, trajectory[ip], trajectory[ip + 1], r1, r2);

                // Account for extra area in the trajectory end; this introduces a small error...
                if (mode == kc2d_ratio)
                    factor *= area / (area - KC2D_PI * r2 * r2);
                if (mode == kc2d_particle)
                    factor *= (area - KC2D_PI * r2 * r2) / area;
            }
            else
                area = kc2d_half_cylinder(&cylinder, trajectory[ip], trajectory[ip + 1], r1, r2);

            kc2d_rasterize_ll(&cylinder, area, grid, lgrid, dims, grid_size, factor, mode);
        }
    }
    
    KC2D_FREE(lgrid);
    KC2D_FREE(trajectory);
}


kc2d_real        kc2d_zero_corner(kc2d_rvec2 pixl, kc2d_rvec2 pixu, kc2d_rvec2 pos, kc2d_real r)
{
    // Particle is outside the pixel
    if ((pos.x <= pixl.x && pos.y <= pixl.y) ||  // Bottom left
        (pos.x >= pixu.x && pos.y <= pixl.y) ||  // Bottom right
        (pos.x >= pixu.x && pos.y >= pixu.y) ||  // Top right
        (pos.x <= pixl.x && pos.y >= pixu.y))    // Top left
        return 0.;

    kc2d_real area = KC2D_PI * r * r;

    // Cut off particle area outside the pixel
    if (pixl.x < pos.x && pos.x < pixu.x)
    {
        area -= kc2d_circular_segment(r, pixl.y - (pos.y - r));     // Bottom
        area -= kc2d_circular_segment(r, (pos.y + r) - pixu.y);     // Top
    }
    if (pixl.y < pos.y && pos.y < pixu.y)
    {
        area -= kc2d_circular_segment(r, (pos.x + r) - pixu.x);     // Right
        area -= kc2d_circular_segment(r, pixl.x - (pos.x - r));     // Left
    }

    return area;
}


#define KC2D_CORNER_TERM_(r, c, p) KC2D_SQRT((r) * (r) - ((c) - (p)) * ((c) - (p)))


kc2d_real       kc2d_one_corner(kc2d_int corners[4], kc2d_rvec2 pixl, kc2d_rvec2 pixu,
                                kc2d_rvec2 pos, kc2d_real r)
{
    // The bounded corner coordinates
    kc2d_real xc = 0, yc = 0;

    // The coords of the two intersections between the circle and the rectangle
    kc2d_real x1 = 0, y1 = 0, x2 = 0, y2 = 0;

    kc2d_real d, h;
    kc2d_real area = 0.;

    // Lower left corner
    if (corners[0] == 1)
    {
        xc = pixl.x;
        yc = pixl.y;

        x1 = pos.x + KC2D_CORNER_TERM_(r, yc, pos.y);
        y1 = yc;

        x2 = xc;
        y2 = pos.y + KC2D_CORNER_TERM_(r, xc, pos.x);
    }

    // Lower right corner
    else if (corners[1] == 1)
    {
        xc = pixu.x;
        yc = pixl.y;

        x1 = xc;
        y1 = pos.y + KC2D_CORNER_TERM_(r, xc, pos.x);

        x2 = pos.x - KC2D_CORNER_TERM_(r, yc, pos.y);
        y2 = yc;
    }

    // Upper right corner
    else if (corners[2] == 1)
    {
        xc = pixu.x;
        yc = pixu.y;

        x1 = pos.x - KC2D_CORNER_TERM_(r, yc, pos.y);
        y1 = yc;

        x2 = xc;
        y2 = pos.y - KC2D_CORNER_TERM_(r, xc, pos.x);
    }

    // Upper left corner
    else if (corners[3] == 1)
    {
        xc = pixl.x;
        yc = pixu.y;

        x1 = pos.x + KC2D_CORNER_TERM_(r, yc, pos.y);
        y1 = yc;

        x2 = xc;
        y2 = pos.y - KC2D_CORNER_TERM_(r, xc, pos.x);
    }

    d = kc2d_dist_pl(pos.x, pos.y, x1, y1, x2, y2);
    h = r - d;

    // Area of the circular segment
    area += kc2d_circular_segment(r, h);

    // Area of the right triangle
    area += kc2d_dist(xc, yc, x1, y1) * kc2d_dist(xc, yc, x2, y2) / 2;

    return area;
}


kc2d_real       kc2d_two_corner(kc2d_int corners[4], kc2d_rvec2 pixl, kc2d_rvec2 pixu,
                                kc2d_rvec2 pos, kc2d_real r)
{
    // The bounded corner coordinates
    kc2d_real xc1 = 0, yc1 = 0, xc2 = 0, yc2 = 0;

    // The coords of the two intersections between the circle and the rectangle
    kc2d_real x1 = 0, y1 = 0, x2 = 0, y2 = 0;

    kc2d_real d, h;
    kc2d_real area = 0.;

    // Bottom corners
    if (corners[0] == 1 && corners[1] == 1)
    {
        xc1 = pixl.x;
        yc1 = pixl.y;

        xc2 = pixu.x;
        yc2 = pixl.y;

        x1 = xc1;
        y1 = pos.y + KC2D_CORNER_TERM_(r, xc1, pos.x);

        x2 = xc2;
        y2 = pos.y + KC2D_CORNER_TERM_(r, xc2, pos.x);
    }

    // Top corners
    else if (corners[2] == 1 && corners[3] == 1)
    {
        xc1 = pixl.x;
        yc1 = pixu.y;

        xc2 = pixu.x;
        yc2 = pixu.y;

        x1 = xc1;
        y1 = pos.y - KC2D_CORNER_TERM_(r, xc1, pos.x);

        x2 = xc2;
        y2 = pos.y - KC2D_CORNER_TERM_(r, xc2, pos.x);
    }

    // Right corners
    else if (corners[1] == 1 && corners[2] == 1)
    {
        xc1 = pixu.x;
        yc1 = pixl.y;

        xc2 = pixu.x;
        yc2 = pixu.y;

        x1 = pos.x - KC2D_CORNER_TERM_(r, yc1, pos.y);
        y1 = yc1;

        x2 = pos.x - KC2D_CORNER_TERM_(r, yc2, pos.y);
        y2 = yc2;
    }

    // Left corners
    else if (corners[0] == 1 && corners[3] == 1)
    {
        xc1 = pixl.x;
        yc1 = pixl.y;

        xc2 = pixl.x;
        yc2 = pixu.y;

        x1 = pos.x + KC2D_CORNER_TERM_(r, yc1, pos.y);
        y1 = yc1;

        x2 = pos.x + KC2D_CORNER_TERM_(r, yc2, pos.y);
        y2 = yc2;
    }

    d = kc2d_dist_pl(pos.x, pos.y, x1, y1, x2, y2);
    h = r - d;

    // Area of the circular segment
    area += kc2d_circular_segment(r, h);

    // Area of triangle1 and triangle2
    area += kc2d_triangle(x1, y1, xc2, yc2, xc1, yc1);
    area += kc2d_triangle(x1, y1, xc2, yc2, x2, y2);

    return area;
}


kc2d_real       kc2d_three_corner(kc2d_int corners[4], kc2d_rvec2 pixl, kc2d_rvec2 pixu,
                                  kc2d_rvec2 pos, kc2d_real r)
{
    kc2d_real        xc1 = 0, yc1 = 0;       // Bounded corner 1
    kc2d_real        xc3 = 0, yc3 = 0;       // Bounded corner 3
    kc2d_real        xc4 = 0, yc4 = 0;       // Unbounded corner

    // The coords of the two intersections between the circle and the rectangle
    kc2d_real x1 = 0, y1 = 0, x2 = 0, y2 = 0;

    kc2d_real area = 0.;

    // Bottom right three corners
    if (corners[0] == 1 && corners[1] == 1 && corners[2] == 1)
    {
        xc1 = pixl.x;
        yc3 = pixu.y;

        xc4 = pixl.x;   // Corner outside
        yc4 = pixu.y;   // Corner outside

        x1 = pos.x - KC2D_CORNER_TERM_(r, yc3, pos.y);
        y1 = yc3;

        x2 = xc1;
        y2 = pos.y + KC2D_CORNER_TERM_(r, xc1, pos.x);
    }

    // Top right three corners
    else if (corners[1] == 1 && corners[2] == 1 && corners[3] == 1)
    {
        yc1 = pixl.y;
        xc3 = pixl.x;

        xc4 = pixl.x;   // Corner outside
        yc4 = pixl.y;   // Corner outside

        x1 = xc3;
        y1 = pos.y - KC2D_CORNER_TERM_(r, xc3, pos.x);

        x2 = pos.x - KC2D_CORNER_TERM_(r, yc1, pos.y);
        y2 = yc1;
    }

    // Top left three corners
    else if (corners[2] == 1 && corners[3] == 1 && corners[0] == 1)
    {
        xc1 = pixu.x;
        yc3 = pixl.y;

        xc4 = pixu.x;   // Corner outside
        yc4 = pixl.y;   // Corner outside

        x1 = xc1;
        y1 = pos.y - KC2D_CORNER_TERM_(r, xc1, pos.x);

        x2 = pos.x + KC2D_CORNER_TERM_(r, yc3, pos.y);
        y2 = yc3;
    }

    // Bottom left three corners
    else if (corners[0] == 1 && corners[1] == 1 && corners[3] == 1)
    {
        yc1 = pixu.y;
        xc3 = pixu.x;

        xc4 = pixu.x;   // Corner outside
        yc4 = pixu.y;   // Corner outside

        x1 = xc3;
        y1 = pos.y + KC2D_CORNER_TERM_(r, xc3, pos.x);

        x2 = pos.x + KC2D_CORNER_TERM_(r, yc1, pos.y);
        y2 = yc1;
    }

    // Area of the rectangle
    area += (pixu.x - pixl.x) * (pixu.y - pixl.y);

    // Area of triangle
    area -= kc2d_dist(x1, y1, xc4, yc4) * kc2d_dist(x2, y2, xc4, yc4) / 2;

    // Area of the circular segment
    area += kc2d_circular_segment(r, r - kc2d_dist_pl(pos.x, pos.y, x1, y1, x2, y2));

    return area;
}


kc2d_real       kc2d_four_corner(kc2d_rvec2 pixl, kc2d_rvec2 pixu)
{
    return (pixu.x - pixl.x) * (pixu.y - pixl.y);
}


void            kc2d_static(kc2d_pixels             *pixels,
                            const kc2d_particles    *particles,
                            const kc2d_mode         mode)
{
    // Some cheap input parameter checks
    if (pixels->dims[0] < 2 || pixels->dims[1] < 2 || particles->num_particles < 1)
    {
        fprintf(stderr, "[ERROR]: The input grid should have at least 2x2 cells, and there should "
                "be at least one particle position.\n\n");
        return;
    }

    // Extract members from `pixels` and `particles`
    kc2d_real        *grid = pixels->grid;
    const kc2d_int   *dims = pixels->dims;
    const kc2d_real  *xlim = pixels->xlim;
    const kc2d_real  *ylim = pixels->ylim;

    const kc2d_rvec2 *positions = (kc2d_rvec2*)(particles->positions);
    const kc2d_real  *radii = particles->radii;
    const kc2d_real  *factors = particles->factors;
    const kc2d_int   num_particles = particles->num_particles;

    // Auxilliaries
    kc2d_int         ip, jp, kp, lp;             // Particle / Pixel indices
    kc2d_rvec2       pos;                        // Particle position
    kc2d_rvec2       pixl, pixu;                 // Pixell lower and upper corner phyisical position

    kc2d_real        r, f;                       // Current radius and factor
    kc2d_real        val;                        // Value to be added onto pixel grid
    kc2d_int         nx[2], ny[2];               // Local pixel range for circle
    kc2d_int         corners[4], num_corners;    // Pixel corners inside circle and count

    // Pixel x and y size
    kc2d_rvec2       size = {{(xlim[1] - xlim[0]) / dims[0], (ylim[1] - ylim[0]) / dims[1]}};
    kc2d_real        pixel_area = size.x * size.y;

    // Rasterize individual circles, analytically
    for (ip = 0; ip < num_particles; ++ip)
    {
        pos = positions[ip];
        r = (radii == NULL ? 0. : radii[ip]);
        f = (factors == NULL ? 1. : factors[ip]);

        if (isnan(pos.x) || isnan(pos.y) || isnan(r) || isnan(f))
            continue;

        // If a particle's radius is zero, treat it as a point
        if (r == 0.)
        {
            if (pos.x == xlim[1])
                nx[0] = dims[0] - 1;
            else
                nx[0] = (kc2d_int)KC2D_FLOOR((pos.x - xlim[0]) / size.x);

            if (pos.y == ylim[1])
                ny[0] = dims[1] - 1;
            else
                ny[0] = (kc2d_int)KC2D_FLOOR((pos.y - ylim[0]) / size.y);

            if (nx[0] >= dims[0] || nx[0] < 0 || ny[0] >= dims[1] || ny[0] < 0)
                continue;

            grid[nx[0] * dims[1] + ny[0]] += f;

            continue;
        }

        // Find range of pixels spanned by the circle
        nx[0] = (kc2d_int)KC2D_FLOOR((pos.x - r - xlim[0]) / size.x);
        nx[1] = (kc2d_int)KC2D_FLOOR((pos.x + r - xlim[0]) / size.x);

        ny[0] = (kc2d_int)KC2D_FLOOR((pos.y - r - ylim[0]) / size.y);
        ny[1] = (kc2d_int)KC2D_FLOOR((pos.y + r - ylim[0]) / size.y);

        // Check pixel indices are within the system bounds and truncate them if not
        if (nx[0] >= dims[0])
            continue;
        else if (nx[0] < 0)
            nx[0] = 0;

        if (nx[1] < 0)
            continue;
        else if (nx[1] >= dims[0])
            nx[1] = dims[0] - 1;

        if (ny[0] >= dims[1])
            continue;
        else if (ny[0] < 0)
            ny[0] = 0;

        if (ny[1] < 0)
            continue;
        else if (ny[1] >= dims[1])
            ny[1] = dims[1] - 1;

        // Iterate over all intersected pixels
        for (jp = nx[0]; jp <= nx[1]; ++jp)
        {
            for (kp = ny[0]; kp <= ny[1]; ++kp)
            {
                // Find the number of pixel corners inside the circle
                num_corners = 0;
                for (lp = 0; lp < 4; ++lp)
                    corners[lp] = 0;

                // Pixel lower and upper corner coordinates
                pixl.x = xlim[0] + jp * size.x;
                pixl.y = ylim[0] + kp * size.y;

                pixu.x = pixl.x + size.x;
                pixu.y = pixl.y + size.y;

                // Lower left corner
                if (kc2d_dist2(pixl.x, pixl.y, pos.x, pos.y) < r * r)
                {
                    num_corners += 1;
                    corners[0] = 1;
                }

                // Lower right corner
                if (kc2d_dist2(pixu.x, pixl.y, pos.x, pos.y) < r * r)
                {
                    num_corners += 1;
                    corners[1] = 1;
                }

                // Upper right corner
                if (kc2d_dist2(pixu.x, pixu.y, pos.x, pos.y) < r * r)
                {
                    num_corners += 1;
                    corners[2] = 1;
                }

                // Upper left corner
                if (kc2d_dist2(pixl.x, pixu.y, pos.x, pos.y) < r * r)
                {
                    num_corners += 1;
                    corners[3] = 1;
                }

                // Compute the intersection area
                if (num_corners == 0)
                    val = kc2d_zero_corner(pixl, pixu, pos, r);
                else if (num_corners == 1)
                    val = kc2d_one_corner(corners, pixl, pixu, pos, r);
                else if (num_corners == 2)
                    val = kc2d_two_corner(corners, pixl, pixu, pos, r);
                else if (num_corners == 3)
                    val = kc2d_three_corner(corners, pixl, pixu, pos, r);
                else
                    val = kc2d_four_corner(pixl, pixu);

                // Truncate `val` to its valid range - necessary due to single precision errors in
                // circular_segment (where it is due to arccos)
                if (val < 0.)
                    val = 0.;
                if (val > pixel_area)
                    val = pixel_area;

                if (val != 0.)
                {
                    if (mode == kc2d_ratio)
                        grid[jp * dims[1] + kp] += f * val / (KC2D_PI * r * r);
                    else if (mode == kc2d_intersection)
                        grid[jp * dims[1] + kp] += f * val;
                    else if (mode == kc2d_particle)
                        grid[jp * dims[1] + kp] += f * (KC2D_PI * r * r);
                    else if (mode == kc2d_one)
                        grid[jp * dims[1] + kp] += f;
                }
            }
        }
    }
}


void            kc2d_rasterize(kc2d_poly            *poly,
                               const kc2d_real      area,
                               const kc2d_real      factor,
                               kc2d_pixels          *pixels,
                               kc2d_real            *local_grid,
                               const kc2d_mode      mode)
{
    // Some cheap input parameter checks
    if (pixels->dims[0] < 2 || pixels->dims[1] < 2)
    {
        fprintf(stderr, "[ERROR]: The input grid should have at least 2x2 cells.\n\n");
        return;
    }

    // Extract members from `pixels` and `particles`
    kc2d_real        *grid = pixels->grid;
    const kc2d_int   *dims = pixels->dims;
    const kc2d_real  *xlim = pixels->xlim;
    const kc2d_real  *ylim = pixels->ylim;

    // Initialise global pixel grid
    kc2d_real        xsize = xlim[1] - xlim[0];
    kc2d_real        ysize = ylim[1] - ylim[0];

    kc2d_rvec2       grid_size = {{xsize / dims[0], ysize / dims[1]}};

    // Translate poly such that the grid origin is at (0, 0)
    kc2d_rvec2       shift = {{-xlim[0], -ylim[0]}};
    kc2d_translate(poly, shift);

    // Local grid which will be used for rasterising
    if (local_grid == NULL)
    {
        local_grid = (kc2d_real*)KC2D_CALLOC((size_t)dims[0] * dims[1], sizeof(kc2d_real));
        kc2d_rasterize_ll(poly, area, grid, local_grid, dims, grid_size, factor, mode);
        KC2D_FREE(local_grid);
    }
    else
        kc2d_rasterize_ll(poly, area, grid, local_grid, dims, grid_size, factor, mode);
}




/* R2D macros and functions. They were taken from https://github.com/devonmpowell/r3d and
 * significantly optimised for `polyorder = 0` (i.e. only area / zeroth moment). */
#define ONE_THIRD 0.333333333333333333333333333333333333333333333333333333
#define ONE_SIXTH 0.16666666666666666666666666666666666666666666666666666667
#define dot(va, vb) (va.x*vb.x + va.y*vb.y)
#define wav(va, wa, vb, wb, vr) {           \
    vr.x = (wa*va.x + wb*vb.x)/(wa + wb);   \
    vr.y = (wa*va.y + wb*vb.y)/(wa + wb);   \
}
#define norm(v) {                   \
    kc2d_real tmplen = sqrt(dot(v, v));  \
    v.x /= (tmplen + 1.0e-299);     \
    v.y /= (tmplen + 1.0e-299);     \
}


void kc2d_init_poly(kc2d_poly* poly, kc2d_rvec2* vertices, kc2d_int numverts) {
    // direct access to vertex buffer
    kc2d_vertex* vertbuffer = poly->verts; 
    kc2d_int* nverts = &poly->nverts; 

    // init the poly
    *nverts = numverts;
    kc2d_int v;
    for(v = 0; v < *nverts; ++v) {
        if (vertices != NULL)
            vertbuffer[v].pos = vertices[v];
        vertbuffer[v].pnbrs[0] = (v+1)%(*nverts);
        vertbuffer[v].pnbrs[1] = (*nverts+v-1)%(*nverts);
    }
}


void kc2d_translate(kc2d_poly* poly, kc2d_rvec2 shift) {
	kc2d_int v;
	for(v = 0; v < poly->nverts; ++v) {
		poly->verts[v].pos.x += shift.x;
		poly->verts[v].pos.y += shift.y;
	}
}


void kc2d_get_ibox(kc2d_poly* poly, kc2d_dvec2 ibox[2], kc2d_rvec2 d) {
    kc2d_int i, v;
    kc2d_rvec2 rbox[2];
    for(i = 0; i < 2; ++i) {
        rbox[0].xy[i] = 1.0e30;
        rbox[1].xy[i] = -1.0e30;
    }
    for(v = 0; v < poly->nverts; ++v) {
        for(i = 0; i < 2; ++i) {
            if(poly->verts[v].pos.xy[i] < rbox[0].xy[i]) rbox[0].xy[i] = poly->verts[v].pos.xy[i];
            if(poly->verts[v].pos.xy[i] > rbox[1].xy[i]) rbox[1].xy[i] = poly->verts[v].pos.xy[i];
        }
    }
    for(i = 0; i < 2; ++i) {
        ibox[0].ij[i] = floor(rbox[0].xy[i]/d.xy[i]);
        ibox[1].ij[i] = ceil(rbox[1].xy[i]/d.xy[i]);
    }
}


void kc2d_clamp_ibox(kc2d_poly* poly, kc2d_dvec2 ibox[2], kc2d_dvec2 clampbox[2], kc2d_rvec2 d) {
    kc2d_int i, nboxclip;
    kc2d_plane boxfaces[4];
    nboxclip = 0;
    memset(boxfaces, 0, sizeof(boxfaces));
    for(i = 0; i < 2; ++i) {
        if(ibox[1].ij[i] <= clampbox[0].ij[i] || ibox[0].ij[i] >= clampbox[1].ij[i]) {
            memset(ibox, 0, 2*sizeof(kc2d_dvec2));
            poly->nverts = 0;
            return;
        }
        if(ibox[0].ij[i] < clampbox[0].ij[i]) {
            ibox[0].ij[i] = clampbox[0].ij[i];
            boxfaces[nboxclip].d = -clampbox[0].ij[i]*d.xy[i];
            boxfaces[nboxclip].n.xy[i] = 1.0;
            nboxclip++;
        }
        if(ibox[1].ij[i] > clampbox[1].ij[i]) {
            ibox[1].ij[i] = clampbox[1].ij[i];
            boxfaces[nboxclip].d = clampbox[1].ij[i]*d.xy[i];
            boxfaces[nboxclip].n.xy[i] = -1.0;
            nboxclip++;
        }   
    }
    if(nboxclip) kc2d_clip(poly, boxfaces, nboxclip);
}


void kc2d_clip(kc2d_poly* poly, kc2d_plane* planes, kc2d_int nplanes) {

    // variable declarations
    kc2d_int v, p, np, onv, vstart, vcur, vnext, numunclipped; 

    // direct access to vertex buffer
    kc2d_vertex* vertbuffer = poly->verts; 
    kc2d_int* nverts = &poly->nverts; 
    if(*nverts <= 0) return;

    // signed distances to the clipping plane
    kc2d_real sdists[KC2D_MAX_VERTS];
    kc2d_real smin, smax;

    // for marking clipped vertices
    kc2d_int clipped[KC2D_MAX_VERTS];

    // loop over each clip plane
    for(p = 0; p < nplanes; ++p) {
    
        // calculate signed distances to the clip plane
        onv = *nverts;
        smin = 1.0e30;
        smax = -1.0e30;
        memset(&clipped, 0, sizeof(clipped));
        for(v = 0; v < onv; ++v) {
            sdists[v] = planes[p].d + dot(vertbuffer[v].pos, planes[p].n);
            if(sdists[v] < smin) smin = sdists[v];
            if(sdists[v] > smax) smax = sdists[v];
            if(sdists[v] < 0.0) clipped[v] = 1;
        }

        // skip this face if the poly lies entirely on one side of it 
        if(smin >= 0.0) continue;
        if(smax <= 0.0) {
            *nverts = 0;
            return;
        }

        // check all edges and insert new vertices on the bisected edges 
        for(vcur = 0; vcur < onv; ++vcur) {
            if(clipped[vcur]) continue;
            for(np = 0; np < 2; ++np) {
                vnext = vertbuffer[vcur].pnbrs[np];
                if(!clipped[vnext]) continue;
                vertbuffer[*nverts].pnbrs[1-np] = vcur;
                vertbuffer[*nverts].pnbrs[np] = -1;
                vertbuffer[vcur].pnbrs[np] = *nverts;
                wav(vertbuffer[vcur].pos, -sdists[vnext],
                    vertbuffer[vnext].pos, sdists[vcur],
                    vertbuffer[*nverts].pos);
                (*nverts)++;
            }
        }

        // for each new vert, search around the poly for its new neighbors
        // and doubly-link everything
        for(vstart = onv; vstart < *nverts; ++vstart) {
            if(vertbuffer[vstart].pnbrs[1] >= 0) continue;
            vcur = vertbuffer[vstart].pnbrs[0];
            do {
                vcur = vertbuffer[vcur].pnbrs[0]; 
            } while(vcur < onv);
            vertbuffer[vstart].pnbrs[1] = vcur;
            vertbuffer[vcur].pnbrs[0] = vstart;
        }

        // go through and compress the vertex list, removing clipped verts
        // and re-indexing accordingly (reusing `clipped` to re-index everything)
        numunclipped = 0;
        for(v = 0; v < *nverts; ++v) {
            if(!clipped[v]) {
                vertbuffer[numunclipped] = vertbuffer[v];
                clipped[v] = numunclipped++;
            }
        }
        *nverts = numunclipped;
        for(v = 0; v < *nverts; ++v) {
            vertbuffer[v].pnbrs[0] = clipped[vertbuffer[v].pnbrs[0]];
            vertbuffer[v].pnbrs[1] = clipped[vertbuffer[v].pnbrs[1]];
        }   
    }
}


void kc2d_reduce(kc2d_poly* poly, kc2d_real* moments) {

    // var declarations
    kc2d_int vcur, vnext;
    kc2d_real twoa;
    kc2d_rvec2 v0, v1;

    // direct access to vertex buffer
    kc2d_vertex* vertbuffer = poly->verts; 
    kc2d_int* nverts = &poly->nverts; 

    // zero the moments
    moments[0] = 0.0;

    if(*nverts <= 0) return;

    // iterate over edges and compute a sum over simplices 
    for(vcur = 0; vcur < *nverts; ++vcur) {

        vnext = vertbuffer[vcur].pnbrs[0];
        v0 = vertbuffer[vcur].pos;
        v1 = vertbuffer[vnext].pos;

        twoa = (v0.x*v1.y - v0.y*v1.x);

        // base case
        moments[0] += 0.5*twoa;
    }
}


void kc2d_split_coord(kc2d_poly* inpoly, kc2d_poly** outpolys, kc2d_real coord, kc2d_int ax) {

    // direct access to vertex buffer
    if(inpoly->nverts <= 0) return;
    kc2d_int* nverts = &inpoly->nverts;
    kc2d_vertex* vertbuffer = inpoly->verts; 
    kc2d_int v, np, onv, vcur, vnext, vstart, nright, cside;
    kc2d_rvec2 newpos;
    kc2d_int side[KC2D_MAX_VERTS];
    kc2d_real sdists[KC2D_MAX_VERTS];

    // calculate signed distances to the clip plane
    nright = 0;
    memset(&side, 0, sizeof(side));
    for(v = 0; v < *nverts; ++v) {
        sdists[v] = vertbuffer[v].pos.xy[ax] - coord;
        if(sdists[v] > 0.0) {
            side[v] = 1;
            nright++;
        }
    }

    // return if the poly lies entirely on one side of it 
    if(nright == 0) {
        *(outpolys[0]) = *inpoly;
        outpolys[1]->nverts = 0;
        return;
    }
    if(nright == *nverts) {
        *(outpolys[1]) = *inpoly;
        outpolys[0]->nverts = 0;
        return;
    }

    // check all edges and insert new vertices on the bisected edges 
    onv = inpoly->nverts;
    for(vcur = 0; vcur < onv; ++vcur) {
        if(side[vcur]) continue;
        for(np = 0; np < 2; ++np) {
            vnext = vertbuffer[vcur].pnbrs[np];
            if(!side[vnext]) continue;
            wav(vertbuffer[vcur].pos, -sdists[vnext],
                vertbuffer[vnext].pos, sdists[vcur],
                newpos);
            vertbuffer[*nverts].pos = newpos;
            vertbuffer[vcur].pnbrs[np] = *nverts;
            vertbuffer[*nverts].pnbrs[np] = -1;
            vertbuffer[*nverts].pnbrs[1-np] = vcur;
            (*nverts)++;
            side[*nverts] = 1;
            vertbuffer[*nverts].pos = newpos;
            vertbuffer[*nverts].pnbrs[1-np] = -1;
            vertbuffer[*nverts].pnbrs[np] = vnext;
            vertbuffer[vnext].pnbrs[1-np] = *nverts;
            (*nverts)++;
        }
    }

    // for each new vert, search around the poly for its new neighbors
    // and doubly-link everything
    for(vstart = onv; vstart < *nverts; ++vstart) {
        if(vertbuffer[vstart].pnbrs[1] >= 0) continue;
        vcur = vertbuffer[vstart].pnbrs[0];
        do {
            vcur = vertbuffer[vcur].pnbrs[0]; 
        } while(vcur < onv);
        vertbuffer[vstart].pnbrs[1] = vcur;
        vertbuffer[vcur].pnbrs[0] = vstart;
    }

    // copy and compress vertices into their new buffers
    // reusing side[] for reindexing
    onv = *nverts;
    outpolys[0]->nverts = 0;
    outpolys[1]->nverts = 0;
    for(v = 0; v < onv; ++v) {
        cside = side[v];
        outpolys[cside]->verts[outpolys[cside]->nverts] = vertbuffer[v];
        side[v] = (outpolys[cside]->nverts)++;
    }

    for(v = 0; v < outpolys[0]->nverts; ++v) 
        for(np = 0; np < 2; ++np)
            outpolys[0]->verts[v].pnbrs[np] = side[outpolys[0]->verts[v].pnbrs[np]];
    for(v = 0; v < outpolys[1]->nverts; ++v) 
        for(np = 0; np < 2; ++np)
            outpolys[1]->verts[v].pnbrs[np] = side[outpolys[1]->verts[v].pnbrs[np]];
}


#define KC2D_STACKSIZE 100


void kc2d_rasterize_local(kc2d_poly* poly, kc2d_dvec2 ibox[2], kc2d_real* dest_grid, kc2d_rvec2 d) {

    kc2d_int i, spax, dmax, nstack, siz;
    kc2d_poly* children[2];
    kc2d_dvec2 gridsz;

    // return if any parameters are bad 
    for(i = 0; i < 2; ++i) gridsz.ij[i] = ibox[1].ij[i]-ibox[0].ij[i];  
    if(!poly || poly->nverts <= 0 || !dest_grid || 
            gridsz.i <= 0 || gridsz.j <= 0) return;
    
    kc2d_real moments;

    // explicit stack-based implementation
    // stack size should never overflow in this implementation, 
    // even for large input grids (up to ~512^2) 
    typedef struct {
        kc2d_poly poly;
        kc2d_dvec2 ibox[2];
    } stack_elem;

    // Small stack optimisation - only heap-allocate if needed_stacksize > KC2D_STACKSIZE
    stack_elem small_stack[KC2D_STACKSIZE];
    stack_elem *stack;

    kc2d_int needed_stacksize = (kc2d_int)(ceil(log2(gridsz.i))+ceil(log2(gridsz.j))+1);
    if (KC2D_STACKSIZE < needed_stacksize)
        stack = (stack_elem*)KC2D_MALLOC(sizeof(stack_elem) * needed_stacksize);
    else
        stack = small_stack;

    // push the original polyhedron onto the stack
    // and recurse until child polyhedra occupy single rasters
    nstack = 0;
    stack[nstack].poly = *poly;
    memcpy(stack[nstack].ibox, ibox, 2*sizeof(kc2d_dvec2));
    nstack++;
    while(nstack > 0) {

        // pop the stack
        // if the leaf is empty, skip it
        --nstack;
        if(stack[nstack].poly.nverts <= 0) continue;
        
        // find the longest axis along which to split 
        dmax = 0;
        spax = 0;
        for(i = 0; i < 2; ++i) {
            siz = stack[nstack].ibox[1].ij[i]-stack[nstack].ibox[0].ij[i];
            if(siz > dmax) {
                dmax = siz; 
                spax = i;
            }   
        }

        // if all three axes are only one raster long, reduce the single raster to the dest grid
#define gind(ii, jj, mm) (((ii-ibox[0].i)*gridsz.j+(jj-ibox[0].j))+mm)
        if(dmax == 1) {
            kc2d_reduce(&stack[nstack].poly, &moments);
            // TODO: cell shifting for accuracy
            dest_grid[gind(stack[nstack].ibox[0].i, stack[nstack].ibox[0].j, 0)] += moments;
            continue;
        }

        // split the poly and push children to the stack
        children[0] = &stack[nstack].poly;
        children[1] = &stack[nstack+1].poly;
        kc2d_split_coord(
            &stack[nstack].poly,
            children,
            d.xy[spax]*(stack[nstack].ibox[0].ij[spax]+dmax/2),
            spax
        );
        memcpy(stack[nstack+1].ibox, stack[nstack].ibox, 2*sizeof(kc2d_dvec2));
        //stack[nstack].ibox[0].ij[spax] += dmax/2;
        //stack[nstack+1].ibox[1].ij[spax] -= dmax-dmax/2; 

        stack[nstack].ibox[1].ij[spax] -= dmax-dmax/2; 
        stack[nstack+1].ibox[0].ij[spax] += dmax/2;

        nstack += 2;
    }

    // If we heap-allocated `stack`, free it
    if (KC2D_STACKSIZE < needed_stacksize)
        KC2D_FREE(stack);
}
