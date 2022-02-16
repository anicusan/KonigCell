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
 * The KonigCell3D core voxellisation routine is built on the R3D library (Powell and Abel, 2015
 * and LA-UR-15-26964). Andrei Leonard Nicusan modified the R3D code in 2021: rasterization was
 * optimised for `polyorder = 0` (i.e. only area / zeroth moment); declarations and definitions
 * from the `r3d.h`, `v3d.h` and `r3d.c` were inlined; the power functions used were specialised
 * for single or double precision; variable-length arrays were removed for portability.
 *
 * All rights for the R3D code go to the original authors of the library, whose copyright notice is
 * included below. A sincere thank you for your work.
 *
 * r3d.h
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


#include "konigcell3d.h"
#include "konigcell3d_generated.h"


#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>




/* Functions from R3D used by KonigCell2D. See https://github.com/devonmpowell/r3d */
void kc3d_get_ibox(kc3d_poly* poly, kc3d_dvec3 ibox[2], kc3d_rvec3 d);
void kc3d_clamp_ibox(kc3d_poly* poly, kc3d_dvec3 ibox[2], kc3d_dvec3 clampbox[2], kc3d_rvec3 d);
int kc3d_clip(kc3d_poly *poly, kc3d_plane *planes, kc3d_int nplanes);
void kc3d_rasterize_local(kc3d_poly* poly, kc3d_dvec3 ibox[2], kc3d_real* dest_grid, kc3d_rvec3 d);
void kc3d_split_coord(kc3d_poly* inpoly, kc3d_poly** outpolys, kc3d_real coord, kc3d_int ax);
void kc3d_reduce(kc3d_poly *poly, kc3d_real *moments);
void kc3d_translate(kc3d_poly *poly, kc3d_rvec3 shift);




#define KC3D_PI 3.14159265358979323846


#ifdef SINGLE_PRECISION
    #define KC3D_SQRT(x) sqrtf(x)
#else
    #define KC3D_SQRT(x) sqrt(x)
#endif


/* R3D macros, taken from https://github.com/devonmpowell/r3d */
#define ONE_THIRD 0.333333333333333333333333333333333333333333333333333333
#define ONE_SIXTH 0.16666666666666666666666666666666666666666666666666666667
#define dot(va, vb) (va.x * vb.x + va.y * vb.y + va.z * vb.z)
#define wav(va, wa, vb, wb, vr)                     \
    {                                               \
        vr.x = (wa * va.x + wb * vb.x) / (wa + wb); \
        vr.y = (wa * va.y + wb * vb.y) / (wa + wb); \
        vr.z = (wa * va.z + wb * vb.z) / (wa + wb); \
    }
#define norm(v)                                     \
    {                                               \
        kc3d_real tmplen = KC3D_SQRT(dot(v, v));    \
        v.x /= (tmplen + 1.0e-299);                 \
        v.y /= (tmplen + 1.0e-299);                 \
        v.z /= (tmplen + 1.0e-299);                 \
    }
#define skew(vx, v)                                 \
    {                                               \
        vx[0].x = 0.;                               \
        vx[0].y = -v.z;                             \
        vx[0].z = v.y;                              \
        vx[1].x = v.z;                              \
        vx[1].y = 0.;                               \
        vx[1].z = -v.x;                             \
        vx[2].x = -v.y;                             \
        vx[2].y = v.x;                              \
        vx[2].z = 0.;                               \
    }




/* Absolute value */
kc3d_real       kc3d_fabs(kc3d_real x)
{
    return (x >= 0 ? x : -x);
}


/* Euclidean distance between two 3D points */
kc3d_real       kc3d_dist(kc3d_rvec3 p1, kc3d_rvec3 p2)
{
    return KC3D_SQRT(
        (p1.x - p2.x) * (p1.x - p2.x) +
        (p1.y - p2.y) * (p1.y - p2.y) +
        (p1.z - p2.z) * (p1.z - p2.z)
    );
}


/**
 * Approximate a 3D sphere as a polygon with `KC3D_SC_NUM_VERTS` vertices. The input `poly` must be
 * pre-allocated; it will be initialised by this function. Returns analytical volume.
 */
kc3d_real       kc3d_sphere(kc3d_poly *poly,
                            const kc3d_rvec3 centre,
                            const kc3d_real radius)
{
    kc3d_rvec3  vertices[KC3D_SC_NUM_VERTS];
    kc3d_dvec3  faceinds[KC3D_SC_NUM_FACES];
    kc3d_int    i;

    // Copy vertices and face indices into mutable buffers
    memcpy(vertices, KC3D_SC_VERTS, KC3D_SC_NUM_VERTS * sizeof(kc3d_rvec3));
    memcpy(faceinds, KC3D_SC_FACES, KC3D_SC_NUM_FACES * sizeof(kc3d_dvec3));

    // Flip top spherical cap
    for (i = KC3D_SC_NUM_VERTS / 2; i < KC3D_SC_NUM_VERTS; ++i)
        vertices[i].z = -vertices[i].z;

    // Expand vertices by radius and translate around centre
    for (i = 0; i < KC3D_SC_NUM_VERTS; ++i)
    {
        vertices[i].x = centre.x + radius * vertices[i].x;
        vertices[i].y = centre.y + radius * vertices[i].y;
        vertices[i].z = centre.z + radius * vertices[i].z;
    }

    // Initialise poly vertices
    kc3d_init_poly_tri(poly, vertices, KC3D_SC_NUM_VERTS,
                             faceinds, KC3D_SC_NUM_FACES);

    return 4. / 3. * KC3D_PI * radius * radius * radius;
}


/**
 * Approximate a 3D spherical cylinder (i.e. the convex hull of two oriented spherical halves)
 * between two points `p1` and `p2` with `KC3D_SC_NUM_VERTS` vertices *without the the second
 * spherical cap's volume*. The input `poly` must be pre-allocated; it will be initialised by this
 * function. Returns the analytical volume.
 */
kc3d_real        kc3d_half_cylinder(kc3d_poly *poly, const kc3d_rvec3 p1, const kc3d_rvec3 p2,
                                    const kc3d_real r1, const kc3d_real r2)
{
    kc3d_rvec3  vertices[KC3D_SC_NUM_VERTS];
    kc3d_dvec3  faceinds[KC3D_SC_NUM_FACES];

    kc3d_rvec3  b, v;                       // 3-vectors
    kc3d_rvec3  vx[3], vx2[3], rot[3];      // row-major 3x3 matrices
    kc3d_real   d;

    kc3d_int    half = KC3D_SC_NUM_VERTS / 2;
    kc3d_int    i, j;

    // Copy vertices and face indices into mutable buffers
    memcpy(vertices, KC3D_SC_VERTS, KC3D_SC_NUM_VERTS * sizeof(kc3d_rvec3));
    memcpy(faceinds, KC3D_SC_FACES, KC3D_SC_NUM_FACES * sizeof(kc3d_dvec3));

    // Expand vertices by r1 (the bottom cap) and r2 (the top cap), then rotate them
    for (i = 0; i < half; ++i)
    {
        vertices[i].x *= r1;
        vertices[i].y *= r1;
        vertices[i].z *= r1;
    }

    for (i = half; i < KC3D_SC_NUM_VERTS; ++i)
    {
        vertices[i].x *= r2;
        vertices[i].y *= r2;
        vertices[i].z *= r2;
    }

    // Unit direction vector between p1 and p2
    b.x = p2.x - p1.x;
    b.y = p2.y - p1.y;
    b.z = p2.z - p1.z;
    norm(b);

    // Cross product between [0, 0, 1] and b
    v.x = -b.y;
    v.y = b.x;
    v.z = 0.;

    // Dot product between [0, 0, 1] and b
    d = b.z;

    // Create skew-symmetric cross-product matrix of v
    skew(vx, v);

    // Matrix-multiply vx by vx
    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
            vx2[i].xyz[j] = vx[i].xyz[0] * vx[0].xyz[j] +
                            vx[i].xyz[1] * vx[1].xyz[j] +
                            vx[i].xyz[2] * vx[2].xyz[j];

    // Create rotation matrix from +Z to p2 - p1 as I3 + vx + vx.vx / (1 + d)
    memset(rot, 0, 9 * sizeof(kc3d_real));
    rot[0].x = 1.;
    rot[1].y = 1.;
    rot[2].z = 1.;

    if (kc3d_fabs(1 + d) > 1e-6)
        for (i = 0; i < 3; ++i)
        {
            rot[i].x += vx[i].x + vx2[i].x / (1 + d);
            rot[i].y += vx[i].y + vx2[i].y / (1 + d);
            rot[i].z += vx[i].z + vx2[i].z / (1 + d);
        }

    // Rotate all vertices; reuse b as a copy of the current vertex
    for (i = 0; i < KC3D_SC_NUM_VERTS; ++i)
    {
        b = vertices[i];

        for (j = 0; j < 3; ++j)
            vertices[i].xyz[j] = rot[j].x * b.x + rot[j].y * b.y + rot[j].z * b.z;
    }

    // Finally, translate bottom and top caps
    for (i = 0; i < half; ++i)
    {
        vertices[i].x += p1.x;
        vertices[i].y += p1.y;
        vertices[i].z += p1.z;
    }

    for (i = half; i < KC3D_SC_NUM_VERTS; ++i)
    {
        vertices[i].x += p2.x;
        vertices[i].y += p2.y;
        vertices[i].z += p2.z;
    }

    // Initialise poly vertices
    kc3d_init_poly_tri(poly, vertices, KC3D_SC_NUM_VERTS,
                             faceinds, KC3D_SC_NUM_FACES);

    return 4. / 6. * KC3D_PI * (r1 * r1 * r1 - r2 * r2 * r2) +                  // Spherical caps
           KC3D_PI / 3 * kc3d_dist(p1, p2) * (r1 * r1 + r1 * r2 + r2 * r2);     // Cylinder / cone
}


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
                              const kc3d_real r2)
{
    kc3d_rvec3  vertices[KC3D_SC_NUM_VERTS];
    kc3d_dvec3  faceinds[KC3D_SC_NUM_FACES];

    kc3d_rvec3  b, v;                       // 3-vectors
    kc3d_rvec3  vx[3], vx2[3], rot[3];      // row-major 3x3 matrices
    kc3d_real   d;

    kc3d_int    half = KC3D_SC_NUM_VERTS / 2;
    kc3d_int    i, j;

    // Copy vertices and face indices into mutable buffers
    memcpy(vertices, KC3D_SC_VERTS, KC3D_SC_NUM_VERTS * sizeof(kc3d_rvec3));
    memcpy(faceinds, KC3D_SC_FACES, KC3D_SC_NUM_FACES * sizeof(kc3d_dvec3));

    // Flip top spherical cap
    for (i = KC3D_SC_NUM_VERTS / 2; i < KC3D_SC_NUM_VERTS; ++i)
        vertices[i].z = -vertices[i].z;

    // Expand vertices by r1 (the bottom cap) and r2 (the top cap), then rotate them
    for (i = 0; i < half; ++i)
    {
        vertices[i].x *= r1;
        vertices[i].y *= r1;
        vertices[i].z *= r1;
    }

    for (i = half; i < KC3D_SC_NUM_VERTS; ++i)
    {
        vertices[i].x *= r2;
        vertices[i].y *= r2;
        vertices[i].z *= r2;
    }

    // Unit direction vector between p1 and p2
    b.x = p2.x - p1.x;
    b.y = p2.y - p1.y;
    b.z = p2.z - p1.z;
    norm(b);

    // Cross product between [0, 0, 1] and b
    v.x = -b.y;
    v.y = b.x;
    v.z = 0.;

    // Dot product between [0, 0, 1] and b
    d = b.z;

    // Create skew-symmetric cross-product matrix of v
    skew(vx, v);

    // Matrix-multiply vx by vx
    for (i = 0; i < 3; ++i)
        for (j = 0; j < 3; ++j)
            vx2[i].xyz[j] = vx[i].xyz[0] * vx[0].xyz[j] +
                            vx[i].xyz[1] * vx[1].xyz[j] +
                            vx[i].xyz[2] * vx[2].xyz[j];

    // Create rotation matrix from +Z to p2 - p1 as I3 + vx + vx.vx / (1 + d)
    memset(rot, 0, 9 * sizeof(kc3d_real));
    rot[0].x = 1.;
    rot[1].y = 1.;
    rot[2].z = 1.;

    if (kc3d_fabs(1 + d) > 1e-6)
        for (i = 0; i < 3; ++i)
        {
            rot[i].x += vx[i].x + vx2[i].x / (1 + d);
            rot[i].y += vx[i].y + vx2[i].y / (1 + d);
            rot[i].z += vx[i].z + vx2[i].z / (1 + d);
        }

    // Rotate all vertices; reuse b as a copy of the current vertex
    for (i = 0; i < KC3D_SC_NUM_VERTS; ++i)
    {
        b = vertices[i];

        for (j = 0; j < 3; ++j)
            vertices[i].xyz[j] = rot[j].x * b.x + rot[j].y * b.y + rot[j].z * b.z;
    }

    // Finally, translate bottom and top caps
    for (i = 0; i < half; ++i)
    {
        vertices[i].x += p1.x;
        vertices[i].y += p1.y;
        vertices[i].z += p1.z;
    }

    for (i = half; i < KC3D_SC_NUM_VERTS; ++i)
    {
        vertices[i].x += p2.x;
        vertices[i].y += p2.y;
        vertices[i].z += p2.z;
    }

    // Initialise poly vertices
    kc3d_init_poly_tri(poly, vertices, KC3D_SC_NUM_VERTS,
                             faceinds, KC3D_SC_NUM_FACES);

    return 4. / 6. * KC3D_PI * (r1 * r1 * r1 + r2 * r2 * r2) +                  // Spherical caps
           KC3D_PI / 3 * kc3d_dist(p1, p2) * (r1 * r1 + r1 * r2 + r2 * r2);     // Cylinder / cone
}


/**
 * Rasterize a polygon `poly` with volume `volume` onto a C-ordered voxel grid `grid` with `dims`
 * dimensions and xyz `grid_size`, using a local `lgrid` for temporary calculations in the voxels
 * spanning the rectangular approximation of the polygon.
 *
 * The area ratio is multiplied by `factor` and *added* onto the global `grid`.
 * The local grid `lgrid` is reinitialised to zero at the end of the function.
 */
void            kc3d_rasterize_ll(kc3d_poly* KC3D_RESTRICT  poly,
                                  kc3d_real                 volume,
                                  kc3d_real* KC3D_RESTRICT  grid,
                                  kc3d_real* KC3D_RESTRICT  lgrid,
                                  const kc3d_int            dims[3],
                                  const kc3d_rvec3          grid_size,
                                  const kc3d_real           factor,
                                  const kc3d_mode           mode)
{
    kc3d_dvec3  clampbox[2] = {{{0, 0, 0}}, {{dims[0], dims[1], dims[2]}}};
    kc3d_dvec3  ibox[2];        // Local grid's range of indices in the global grid
    kc3d_int    lx, ly, lz;     // Local grid's written number of rows and columns
    kc3d_int    i, j, k;        // Iterators

    // Find the range of indices spanned by `poly`, then clamp them if `poly` extends out of `grid`
    kc3d_get_ibox(poly, ibox, grid_size);
    kc3d_clamp_ibox(poly, ibox, clampbox, grid_size);

    // Initialise local grid for the voxellisation step
    lx = ibox[1].i - ibox[0].i;
    ly = ibox[1].j - ibox[0].j;
    lz = ibox[1].k - ibox[0].k;

    // Rasterize the polygon onto the local grid and compute the total area occupied by `poly`
    kc3d_rasterize_local(poly, ibox, lgrid, grid_size);

#define KC3D_GIDX (i * dims[1] * dims[2] + j * dims[2] + k)
#define KC3D_LIDX ((i - ibox[0].i) * ly * lz + (j - ibox[0].j) * lz + (k - ibox[0].k))

    // Add values from the local grid to the global one, depending on the voxellisation `mode`
    if (mode == kc3d_ratio)
    {
        for (i = ibox[0].i; i < ibox[1].i; ++i)
            for (j = ibox[0].j; j < ibox[1].j; ++j)
                for (k = ibox[0].k; k < ibox[1].k; ++k)
                    grid[KC3D_GIDX] += factor * lgrid[KC3D_LIDX] / volume;
    }

    else if (mode == kc3d_intersection)
    {
        for (i = ibox[0].i; i < ibox[1].i; ++i)
            for (j = ibox[0].j; j < ibox[1].j; ++j)
                for (k = ibox[0].k; k < ibox[1].k; ++k)
                    grid[KC3D_GIDX] += factor * lgrid[KC3D_LIDX];
    }

    else if (mode == kc3d_particle)
    {
        for (i = ibox[0].i; i < ibox[1].i; ++i)
            for (j = ibox[0].j; j < ibox[1].j; ++j)
                for (k = ibox[0].k; k < ibox[1].k; ++k)
                    if (lgrid[KC3D_LIDX] != 0.)
                        grid[KC3D_GIDX] += factor * volume;
    }

    else if (mode == kc3d_one)
    {
        for (i = ibox[0].i; i < ibox[1].i; ++i)
            for (j = ibox[0].j; j < ibox[1].j; ++j)
                for (k = ibox[0].k; k < ibox[1].k; ++k)
                    if (lgrid[KC3D_LIDX] != 0.)
                        grid[KC3D_GIDX] += factor;
    }

#undef KC3D_GIDX
#undef KC3D_LIDX

    // Reinitialise the written local grid to zero
    for (i = 0; i < lx * ly * lz; ++i)
        lgrid[i] = 0.;
}


/* Check if the particle position at index `ip` is valid. */
kc3d_int        kc3d_valid_position(const kc3d_particles *particles, const kc3d_int ip)
{
    // Extract attributes needed
    kc3d_real   x = particles->positions[3 * ip];
    kc3d_real   y = particles->positions[3 * ip + 1];
    kc3d_real   z = particles->positions[3 * ip + 2];
    kc3d_real   r = (particles->radii == NULL ? 1e-6 : particles->radii[ip]);
    kc3d_real   f = (particles->factors == NULL ? 1. : particles->factors[ip]);

    return !(isnan(x) || isnan(y) || isnan(z) || isnan(r) || isnan(f));
}


/* Find next valid trajectory's starting index, after `start`. */
kc3d_int        kc3d_next_segment_start(const kc3d_particles    *particles,
                                        const kc3d_int          start)
{
    kc3d_int    ip = start;

    while (ip < particles->num_particles)
    {
        // Segments must have at least two valid positions
        if (ip < particles->num_particles - 1 && kc3d_valid_position(particles, ip) &&
                kc3d_valid_position(particles, ip + 1))
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
kc3d_int        kc3d_next_segment(kc3d_int              *segment_bounds,
                                  const kc3d_particles  *particles,
                                  const kc3d_int        start)
{
    kc3d_int    ip;

    // Find starting index (inclusive)
    ip = kc3d_next_segment_start(particles, start);
    if (ip >= particles->num_particles)
        return 0;

    segment_bounds[0] = ip;

    // Find ending index (exclusive)
    ip = segment_bounds[0] + 2;         // Already checked next one is valid too
    while (ip < particles->num_particles && kc3d_valid_position(particles, ip))
        ++ip;

    segment_bounds[1] = ip;

    // Find next segment's start
    return kc3d_next_segment_start(particles, ip);
}



/**
 * Compute the occupancy grid of a single circular *moving* particle's trajectory.
 *
 * This corresponds to the voxellisation of moving circular particles, such that for every two
 * consecutive particle locations, a 2D cylinder (i.e. convex hull of two circles at the two
 * particle positions), the fraction of its area that intersets a voxel is multiplied with the
 * time between the two particle locations and saved in the input `voxels`.
 */
void            kc3d_dynamic(kc3d_voxels            *voxels,
                             const kc3d_particles   *particles,
                             const kc3d_mode        mode,
                             const kc3d_int         omit_last)
{
    // Some cheap input parameter checks
    if (voxels->dims[0] < 2 || voxels->dims[1] < 2 || voxels->dims[2] < 2 ||
        particles->num_particles < 2)
    {
        fprintf(stderr, "[ERROR]: The input grid should have at least 2x2x2 cells, and there "
                "should be at least two particle positions.\n\n");
        return;
    }

    // Extract members from `voxels` and `particles`
    kc3d_real       *grid = voxels->grid;
    const kc3d_int  *dims = voxels->dims;
    const kc3d_real *xlim = voxels->xlim;
    const kc3d_real *ylim = voxels->ylim;
    const kc3d_real *zlim = voxels->zlim;

    const kc3d_real *positions = particles->positions;
    const kc3d_real *radii = particles->radii;
    const kc3d_real *factors = particles->factors;
    const kc3d_int  num_particles = particles->num_particles;

    // Current trajectory segment bounds indices: start (inclusive), end (exclusive)
    kc3d_int        segment_bounds[2];
    kc3d_int        next = 0;

    // Auxilliaries
    kc3d_int        ip;             // Trajectory particle index
    kc3d_real       r1, r2;         // Radii for two particle
    kc3d_real       volume;         // Total area for one 2D cylinder
    kc3d_real       factor;         // Current factor to multiply raster with

    // Initialise global voxel grid
    kc3d_real       xsize = xlim[1] - xlim[0];
    kc3d_real       ysize = ylim[1] - ylim[0];
    kc3d_real       zsize = zlim[1] - zlim[0];

    kc3d_rvec3      grid_size = {{xsize / dims[0], ysize / dims[1], zsize / dims[2]}};
    kc3d_real       temp = (grid_size.x < grid_size.y ? grid_size.x : grid_size.y);
    kc3d_real       rsmall = 1.0e-6 * (temp < grid_size.z ? temp : grid_size.z);

    // Local grid which will be used for rasterising
    kc3d_real       *lgrid = (kc3d_real*)KC3D_CALLOC((size_t)dims[0] * dims[1] * dims[2],
                                                     sizeof(kc3d_real));

    // Polygonal shapes used for the particle trajectories
    kc3d_poly        cylinder;

    // Copy `positions` to new local array and translate them such that the grid origin is (0, 0, 0)
    kc3d_rvec3      *trajectory = (kc3d_rvec3*)KC3D_MALLOC(sizeof(kc3d_rvec3) * num_particles);

    for (ip = 0; ip < num_particles; ++ip)
    {
        trajectory[ip].x = positions[3 * ip] - xlim[0];
        trajectory[ip].y = positions[3 * ip + 1] - ylim[0];
        trajectory[ip].z = positions[3 * ip + 2] - zlim[0];
    }

    // Rasterize particle trajectory segments: for each segment, across each two consecutive
    // particle positions, create a polygonal approximation of the convex hull of the two particle
    // locations, minus the second sphere's area (which is added in the previous iteration)
    //
    // Find the next trajectory segment's index bounds and return the future one's start index
    while ((next = kc3d_next_segment(segment_bounds, particles, next)))
    {
        for (ip = segment_bounds[0]; ip < segment_bounds[1] - 1; ++ip)
        {
            r1 = (radii == NULL ? rsmall : radii[ip]);
            r2 = (radii == NULL ? rsmall : radii[ip + 1]);
            factor = (factors == NULL ? 1 : factors[ip]);

            // If this is the last cylinder from a segment, pixellise full cylinder, including
            // spherical cap - unless it's the last segment and omit_last
            if (ip == segment_bounds[1] - 2 && !(next >= num_particles - 1 && omit_last))
            {
                volume = kc3d_cylinder(&cylinder, trajectory[ip], trajectory[ip + 1], r1, r2);

                // Account for extra area in the trajectory end; this introduces a small error...
                if (mode == kc3d_ratio)
                    factor *= volume / (volume - 4. / 3. * KC3D_PI * r2 * r2 * r2);
                if (mode == kc3d_particle)
                    factor *= (volume - 4. / 3. * KC3D_PI * r2 * r2 * r2) / volume;
            }
            else
                volume = kc3d_half_cylinder(&cylinder, trajectory[ip], trajectory[ip + 1], r1, r2);

            kc3d_rasterize_ll(&cylinder, volume, grid, lgrid, dims, grid_size, factor, mode);
        }
    }

    KC3D_FREE(lgrid);
    KC3D_FREE(trajectory);
}


void            kc3d_static(kc3d_voxels             *voxels,
                            const kc3d_particles    *particles,
                            const kc3d_mode         mode)
{
    // Some cheap input parameter checks
    if (voxels->dims[0] < 2 || voxels->dims[1] < 2 || voxels->dims[2] < 2 ||
        particles->num_particles < 2)
    {
        fprintf(stderr, "[ERROR]: The input grid should have at least 2x2x2 cells, and there "
                "should be at least two particle positions.\n\n");
        return;
    }

    // Extract members from `voxels` and `particles`
    kc3d_real        *grid = voxels->grid;
    const kc3d_int   *dims = voxels->dims;
    const kc3d_real  *xlim = voxels->xlim;
    const kc3d_real  *ylim = voxels->ylim;
    const kc3d_real  *zlim = voxels->zlim;

    const kc3d_real  *positions = particles->positions;
    const kc3d_real  *radii = particles->radii;
    const kc3d_real  *factors = particles->factors;
    const kc3d_int   num_particles = particles->num_particles;

    // Auxilliaries
    kc3d_int        ip;             // Trajectory particle index
    kc3d_real       radius;         // Particle radius
    kc3d_real       volume;         // Total area for one 2D cylinder
    kc3d_real       factor;         // Current factor to multiply raster with

    // Initialise global voxel grid
    kc3d_real       xsize = xlim[1] - xlim[0];
    kc3d_real       ysize = ylim[1] - ylim[0];
    kc3d_real       zsize = zlim[1] - zlim[0];

    kc3d_rvec3      grid_size = {{xsize / dims[0], ysize / dims[1], zsize / dims[2]}};
    kc3d_real       temp = (grid_size.x < grid_size.y ? grid_size.x : grid_size.y);
    kc3d_real       rsmall = 1.0e-6 * (temp < grid_size.z ? temp : grid_size.z);

    // Local grid which will be used for rasterising
    kc3d_real       *lgrid = (kc3d_real*)KC3D_CALLOC((size_t)dims[0] * dims[1] * dims[2],
                                                     sizeof(kc3d_real));

    // Polygonal shapes used for the particle trajectories
    kc3d_poly        sphere;

    // Copy `positions` to new local array and translate them such that the grid origin is (0, 0, 0)
    kc3d_rvec3      *trajectory = (kc3d_rvec3*)KC3D_MALLOC(sizeof(kc3d_rvec3) * num_particles);

    for (ip = 0; ip < num_particles; ++ip)
    {
        trajectory[ip].x = positions[3 * ip] - xlim[0];
        trajectory[ip].y = positions[3 * ip + 1] - ylim[0];
        trajectory[ip].z = positions[3 * ip + 2] - zlim[0];
    }

    // Rasterize particle trajectories: create a polygonal approximation of the convex hull of the
    // two particle locations, minus the second circle's area (was added in the previous iteration)
    for (ip = 0; ip < num_particles; ++ip)
    {
        // Skip NaNs - useful for jumping over different trajectories
        if (isnan(trajectory[ip].x) || isnan(trajectory[ip].y) || isnan(trajectory[ip].z))
            continue;

        radius = (radii == NULL ? rsmall : radii[ip]);
        factor = (factors == NULL ? 1 : factors[ip]);

        if (isnan(radius) || isnan(factor))
            continue;

        volume = kc3d_sphere(&sphere, trajectory[ip], radius);
        kc3d_rasterize_ll(&sphere, volume, grid, lgrid, dims, grid_size, factor, mode);
    }

    KC3D_FREE(lgrid);
    KC3D_FREE(trajectory);
}


void            kc3d_rasterize(kc3d_poly            *poly,
                               const kc3d_real      volume,
                               const kc3d_real      factor,
                               kc3d_voxels          *voxels,
                               kc3d_real            *local_grid,
                               const kc3d_mode      mode)
{
    // Some cheap input parameter checks
    if (voxels->dims[0] < 2 || voxels->dims[1] < 2 || voxels->dims[2] < 2)
    {
        fprintf(stderr, "[ERROR]: The input grid should have at least 2x2x2 cells\n\n");
        return;
    }

    // Extract members from `voxels` and `particles`
    kc3d_real        *grid = voxels->grid;
    const kc3d_int   *dims = voxels->dims;
    const kc3d_real  *xlim = voxels->xlim;
    const kc3d_real  *ylim = voxels->ylim;
    const kc3d_real  *zlim = voxels->zlim;

    // Initialise global voxel grid
    kc3d_real       xsize = xlim[1] - xlim[0];
    kc3d_real       ysize = ylim[1] - ylim[0];
    kc3d_real       zsize = zlim[1] - zlim[0];

    kc3d_rvec3      grid_size = {{xsize / dims[0], ysize / dims[1], zsize / dims[2]}};

    // Translate poly such that the grid origin is at (0, 0)
    kc3d_rvec3      shift = {{-xlim[0], -ylim[0], -zlim[0]}};
    kc3d_translate(poly, shift);

    // Local grid which will be used for rasterising
    if (local_grid == NULL)
    {
        local_grid = (kc3d_real*)KC3D_CALLOC((size_t)dims[0] * dims[1] * dims[2],
                                             sizeof(kc3d_real));
        kc3d_rasterize_ll(poly, volume, grid, local_grid, dims, grid_size, factor, mode);
        KC3D_FREE(local_grid);
    }
    else
        kc3d_rasterize_ll(poly, volume, grid, local_grid, dims, grid_size, factor, mode);
}

 
int kc3d_init_poly(kc3d_poly *poly, kc3d_rvec3 *vertices, kc3d_int numverts,
                                     kc3d_int **faceinds, kc3d_int *numvertsperface,
                                     kc3d_int numfaces) {
    // dummy vars
    kc3d_int v, vprev, vcur, vnext, f, np;

    if (numverts > KC3D_MAX_VERTS) {
        #if !defined(NDEBUG)
            fprintf(stderr, "kc3d_init_poly: Max vertex buffer size exceeded");
        #endif
        return 0;
    }
        
    // direct access to vertex buffer
    kc3d_vertex *vertbuffer = poly->verts;
    kc3d_int *nverts = &poly->nverts;        
        
    // count up the number of faces per vertex and act accordingly
    kc3d_int eperv[KC3D_MAX_VERTS];
    kc3d_int minvperf = KC3D_MAX_VERTS;
    kc3d_int maxvperf = 0;
    memset(&eperv, 0, sizeof(eperv));
    for (f = 0; f < numfaces; ++f)
        for (v = 0; v < numvertsperface[f]; ++v) ++eperv[faceinds[f][v]];
    for (v = 0; v < numverts; ++v) {
        if (eperv[v] < minvperf) minvperf = eperv[v];
        if (eperv[v] > maxvperf) maxvperf = eperv[v];
    }

    // clear the poly
    *nverts = 0;

    // return if we were given an invalid poly
    if (minvperf < 3) return 0;

    if (maxvperf == 3) {
        // simple case with no need for duplicate vertices

        // read in vertex locations
        *nverts = numverts;
        for (v = 0; v < *nverts; ++v) {
            vertbuffer[v].pos = vertices[v];
            for (np = 0; np < 3; ++np) vertbuffer[v].pnbrs[np] = KC3D_MAX_VERTS;
        }

        // build graph connectivity by correctly orienting half-edges for each
        // vertex
        for (f = 0; f < numfaces; ++f) {
            for (v = 0; v < numvertsperface[f]; ++v) {
                vprev = faceinds[f][v];
                vcur = faceinds[f][(v + 1) % numvertsperface[f]];
                vnext = faceinds[f][(v + 2) % numvertsperface[f]];
                for (np = 0; np < 3; ++np) {
                    if (vertbuffer[vcur].pnbrs[np] == vprev) {
                        vertbuffer[vcur].pnbrs[(np + 2) % 3] = vnext;
                        break;
                    } else if (vertbuffer[vcur].pnbrs[np] == vnext) {
                        vertbuffer[vcur].pnbrs[(np + 1) % 3] = vprev;
                        break;
                    }
                }
                if (np == 3) {
                    vertbuffer[vcur].pnbrs[1] = vprev;
                    vertbuffer[vcur].pnbrs[0] = vnext;
                }
            }
        }
    } else {
        // we need to create duplicate, degenerate vertices to account for more than
        // three edges per vertex. This is complicated.

        kc3d_int tface = 0;
        for (v = 0; v < numverts; ++v) tface += eperv[v];

        // need more variables
        kc3d_int v0, v1, v00, v11, numunclipped;

        // we need a few extra buffers to handle the necessary operations
        kc3d_vertex vbtmp[3 * KC3D_MAX_VERTS];
        kc3d_int util[3 * KC3D_MAX_VERTS];
        kc3d_int vstart[KC3D_MAX_VERTS];

        // build vertex mappings to degenerate duplicates and read in vertex
        // locations
        *nverts = 0;
        for (v = 0; v < numverts; ++v) {
            if ((*nverts) + eperv[v] > KC3D_MAX_VERTS) {
                #if !defined(NDEBUG)
                    fprintf(stderr, "kc3d_init_poly: Max vertex buffer size exceeded");
                #endif
                return 0;
            }
        
            vstart[v] = *nverts;
            for (vcur = 0; vcur < eperv[v]; ++vcur) {
                vbtmp[*nverts].pos = vertices[v];
                for (np = 0; np < 3; ++np) vbtmp[*nverts].pnbrs[np] = KC3D_MAX_VERTS;
                ++(*nverts);
            }
        }

        // fill in connectivity for all duplicates
        memset(&util, 0, sizeof(util));
        for (f = 0; f < numfaces; ++f) {
            for (v = 0; v < numvertsperface[f]; ++v) {
                vprev = faceinds[f][v];
                vcur = faceinds[f][(v + 1) % numvertsperface[f]];
                vnext = faceinds[f][(v + 2) % numvertsperface[f]];
                kc3d_int vcur_old = vcur;
                vcur = vstart[vcur] + util[vcur];
                util[vcur_old]++;
                vbtmp[vcur].pnbrs[1] = vnext;
                vbtmp[vcur].pnbrs[2] = vprev;
            }
        }

        // link degenerate duplicates, putting them in the correct order use util to
        // mark and avoid double-processing verts
        memset(&util, 0, sizeof(util));
        for (v = 0; v < numverts; ++v) {
            for (v0 = vstart[v]; v0 < vstart[v] + eperv[v]; ++v0) {
                for (v1 = vstart[v]; v1 < vstart[v] + eperv[v]; ++v1) {
                    if (vbtmp[v0].pnbrs[2] == vbtmp[v1].pnbrs[1] && !util[v0]) {
                        vbtmp[v0].pnbrs[2] = v1;
                        vbtmp[v1].pnbrs[0] = v0;
                        util[v0] = 1;
                    }
                }
            }
        }

        // complete vertex pairs
        memset(&util, 0, sizeof(util));
        for (v0 = 0; v0 < numverts; ++v0)
            for (v1 = v0 + 1; v1 < numverts; ++v1) {
                for (v00 = vstart[v0]; v00 < vstart[v0] + eperv[v0]; ++v00)
                    for (v11 = vstart[v1]; v11 < vstart[v1] + eperv[v1]; ++v11) {
                        if (vbtmp[v00].pnbrs[1] == v1 && vbtmp[v11].pnbrs[1] == v0 &&
                                !util[v00] && !util[v11]) {
                            vbtmp[v00].pnbrs[1] = v11;
                            vbtmp[v11].pnbrs[1] = v00;
                            util[v00] = 1;
                            util[v11] = 1;
                        }
                    }
            }

        // remove unnecessary dummy vertices
        memset(&util, 0, sizeof(util));
        for (v = 0; v < numverts; ++v) {
            v0 = vstart[v];
            v1 = vbtmp[v0].pnbrs[0];
            v00 = vbtmp[v0].pnbrs[2];
            v11 = vbtmp[v1].pnbrs[0];
            vbtmp[v00].pnbrs[0] = vbtmp[v0].pnbrs[1];
            vbtmp[v11].pnbrs[2] = vbtmp[v1].pnbrs[1];
            for (np = 0; np < 3; ++np)
                if (vbtmp[vbtmp[v0].pnbrs[1]].pnbrs[np] == v0) break;
            vbtmp[vbtmp[v0].pnbrs[1]].pnbrs[np] = v00;
            for (np = 0; np < 3; ++np)
                if (vbtmp[vbtmp[v1].pnbrs[1]].pnbrs[np] == v1) break;
            vbtmp[vbtmp[v1].pnbrs[1]].pnbrs[np] = v11;
            util[v0] = 1;
            util[v1] = 1;
        }

        // copy to the real vertbuffer and compress
        numunclipped = 0;
        for (v = 0; v < *nverts; ++v) {
            if (!util[v]) {
                vertbuffer[numunclipped] = vbtmp[v];
                util[v] = numunclipped++;
            }
        }
        *nverts = numunclipped;
        for (v = 0; v < *nverts; ++v)
            for (np = 0; np < 3; ++np)
                vertbuffer[v].pnbrs[np] = util[vertbuffer[v].pnbrs[np]];
    }

    return 1;
}


/* Like kc3d_init_poly, but assuming every face has exactly 3 indices */
int kc3d_init_poly_tri(kc3d_poly *poly, kc3d_rvec3 *vertices, kc3d_int numverts,
                                        kc3d_dvec3 *faceinds, kc3d_int numfaces) {
    // dummy vars
    kc3d_int v, vprev, vcur, vnext, f, np;

    if (numverts > KC3D_MAX_VERTS) {
        #if !defined(NDEBUG)
            fprintf(stderr, "kc3d_init_poly: Max vertex buffer size exceeded");
        #endif
        return 0;
    }
        
    // direct access to vertex buffer
    kc3d_vertex *vertbuffer = poly->verts;
    kc3d_int *nverts = &poly->nverts;        
        
    // count up the number of faces per vertex and act accordingly
    kc3d_int eperv[KC3D_MAX_VERTS];
    kc3d_int minvperf = KC3D_MAX_VERTS;
    kc3d_int maxvperf = 0;
    memset(&eperv, 0, sizeof(eperv));
    for (f = 0; f < numfaces; ++f)
        for (v = 0; v < 3; ++v) ++eperv[faceinds[f].ijk[v]];
    for (v = 0; v < numverts; ++v) {
        if (eperv[v] < minvperf) minvperf = eperv[v];
        if (eperv[v] > maxvperf) maxvperf = eperv[v];
    }

    // clear the poly
    *nverts = 0;

    // return if we were given an invalid poly
    if (minvperf < 3) return 0;

    if (maxvperf == 3) {
        // simple case with no need for duplicate vertices

        // read in vertex locations
        *nverts = numverts;
        for (v = 0; v < *nverts; ++v) {
            vertbuffer[v].pos = vertices[v];
            for (np = 0; np < 3; ++np) vertbuffer[v].pnbrs[np] = KC3D_MAX_VERTS;
        }

        // build graph connectivity by correctly orienting half-edges for each
        // vertex
        for (f = 0; f < numfaces; ++f) {
            for (v = 0; v < 3; ++v) {
                vprev = faceinds[f].ijk[v];
                vcur = faceinds[f].ijk[(v + 1) % 3];
                vnext = faceinds[f].ijk[(v + 2) % 3];
                for (np = 0; np < 3; ++np) {
                    if (vertbuffer[vcur].pnbrs[np] == vprev) {
                        vertbuffer[vcur].pnbrs[(np + 2) % 3] = vnext;
                        break;
                    } else if (vertbuffer[vcur].pnbrs[np] == vnext) {
                        vertbuffer[vcur].pnbrs[(np + 1) % 3] = vprev;
                        break;
                    }
                }
                if (np == 3) {
                    vertbuffer[vcur].pnbrs[1] = vprev;
                    vertbuffer[vcur].pnbrs[0] = vnext;
                }
            }
        }
    } else {
        // we need to create duplicate, degenerate vertices to account for more than
        // three edges per vertex. This is complicated.

        kc3d_int tface = 0;
        for (v = 0; v < numverts; ++v) tface += eperv[v];

        // need more variables
        kc3d_int v0, v1, v00, v11, numunclipped;

        // we need a few extra buffers to handle the necessary operations
        kc3d_vertex vbtmp[3 * KC3D_MAX_VERTS];
        kc3d_int util[3 * KC3D_MAX_VERTS];
        kc3d_int vstart[KC3D_MAX_VERTS];

        // build vertex mappings to degenerate duplicates and read in vertex
        // locations
        *nverts = 0;
        for (v = 0; v < numverts; ++v) {
            if ((*nverts) + eperv[v] > KC3D_MAX_VERTS) {
                #if !defined(NDEBUG)
                    fprintf(stderr, "kc3d_init_poly: Max vertex buffer size exceeded");
                #endif
                return 0;
            }
        
            vstart[v] = *nverts;
            for (vcur = 0; vcur < eperv[v]; ++vcur) {
                vbtmp[*nverts].pos = vertices[v];
                for (np = 0; np < 3; ++np) vbtmp[*nverts].pnbrs[np] = KC3D_MAX_VERTS;
                ++(*nverts);
            }
        }

        // fill in connectivity for all duplicates
        memset(&util, 0, sizeof(util));
        for (f = 0; f < numfaces; ++f) {
            for (v = 0; v < 3; ++v) {
                vprev = faceinds[f].ijk[v];
                vcur = faceinds[f].ijk[(v + 1) % 3];
                vnext = faceinds[f].ijk[(v + 2) % 3];
                kc3d_int vcur_old = vcur;
                vcur = vstart[vcur] + util[vcur];
                util[vcur_old]++;
                vbtmp[vcur].pnbrs[1] = vnext;
                vbtmp[vcur].pnbrs[2] = vprev;
            }
        }

        // link degenerate duplicates, putting them in the correct order use util to
        // mark and avoid double-processing verts
        memset(&util, 0, sizeof(util));
        for (v = 0; v < numverts; ++v) {
            for (v0 = vstart[v]; v0 < vstart[v] + eperv[v]; ++v0) {
                for (v1 = vstart[v]; v1 < vstart[v] + eperv[v]; ++v1) {
                    if (vbtmp[v0].pnbrs[2] == vbtmp[v1].pnbrs[1] && !util[v0]) {
                        vbtmp[v0].pnbrs[2] = v1;
                        vbtmp[v1].pnbrs[0] = v0;
                        util[v0] = 1;
                    }
                }
            }
        }

        // complete vertex pairs
        memset(&util, 0, sizeof(util));
        for (v0 = 0; v0 < numverts; ++v0)
            for (v1 = v0 + 1; v1 < numverts; ++v1) {
                for (v00 = vstart[v0]; v00 < vstart[v0] + eperv[v0]; ++v00)
                    for (v11 = vstart[v1]; v11 < vstart[v1] + eperv[v1]; ++v11) {
                        if (vbtmp[v00].pnbrs[1] == v1 && vbtmp[v11].pnbrs[1] == v0 &&
                                !util[v00] && !util[v11]) {
                            vbtmp[v00].pnbrs[1] = v11;
                            vbtmp[v11].pnbrs[1] = v00;
                            util[v00] = 1;
                            util[v11] = 1;
                        }
                    }
            }

        // remove unnecessary dummy vertices
        memset(&util, 0, sizeof(util));
        for (v = 0; v < numverts; ++v) {
            v0 = vstart[v];
            v1 = vbtmp[v0].pnbrs[0];
            v00 = vbtmp[v0].pnbrs[2];
            v11 = vbtmp[v1].pnbrs[0];
            vbtmp[v00].pnbrs[0] = vbtmp[v0].pnbrs[1];
            vbtmp[v11].pnbrs[2] = vbtmp[v1].pnbrs[1];
            for (np = 0; np < 3; ++np)
                if (vbtmp[vbtmp[v0].pnbrs[1]].pnbrs[np] == v0) break;
            vbtmp[vbtmp[v0].pnbrs[1]].pnbrs[np] = v00;
            for (np = 0; np < 3; ++np)
                if (vbtmp[vbtmp[v1].pnbrs[1]].pnbrs[np] == v1) break;
            vbtmp[vbtmp[v1].pnbrs[1]].pnbrs[np] = v11;
            util[v0] = 1;
            util[v1] = 1;
        }

        // copy to the real vertbuffer and compress
        numunclipped = 0;
        for (v = 0; v < *nverts; ++v) {
            if (!util[v]) {
                vertbuffer[numunclipped] = vbtmp[v];
                util[v] = numunclipped++;
            }
        }
        *nverts = numunclipped;
        for (v = 0; v < *nverts; ++v)
            for (np = 0; np < 3; ++np)
                vertbuffer[v].pnbrs[np] = util[vertbuffer[v].pnbrs[np]];
    }

        return 1;
}


void kc3d_get_ibox(kc3d_poly* poly, kc3d_dvec3 ibox[2], kc3d_rvec3 d) {
    kc3d_int i, v;
    kc3d_rvec3 rbox[2];
    for(i = 0; i < 3; ++i) {
        rbox[0].xyz[i] = 1.0e30;
        rbox[1].xyz[i] = -1.0e30;
    }
    for(v = 0; v < poly->nverts; ++v) {
        for(i = 0; i < 3; ++i) {
            if(poly->verts[v].pos.xyz[i] < rbox[0].xyz[i]) rbox[0].xyz[i] = poly->verts[v].pos.xyz[i];
            if(poly->verts[v].pos.xyz[i] > rbox[1].xyz[i]) rbox[1].xyz[i] = poly->verts[v].pos.xyz[i];
        }
    }
    for(i = 0; i < 3; ++i) {
        ibox[0].ijk[i] = floor(rbox[0].xyz[i]/d.xyz[i]);
        ibox[1].ijk[i] = ceil(rbox[1].xyz[i]/d.xyz[i]);
    }
}

void kc3d_clamp_ibox(kc3d_poly* poly, kc3d_dvec3 ibox[2], kc3d_dvec3 clampbox[2], kc3d_rvec3 d) {
    kc3d_int i, nboxclip;
    kc3d_plane boxfaces[6];
    nboxclip = 0;
    memset(boxfaces, 0, sizeof(boxfaces));
    for(i = 0; i < 3; ++i) {
        if(ibox[1].ijk[i] <= clampbox[0].ijk[i] || ibox[0].ijk[i] >= clampbox[1].ijk[i]) {
            memset(ibox, 0, 2*sizeof(kc3d_dvec3));
            poly->nverts = 0;
            return;
        }
        if(ibox[0].ijk[i] < clampbox[0].ijk[i]) {
            ibox[0].ijk[i] = clampbox[0].ijk[i];
            boxfaces[nboxclip].d = -clampbox[0].ijk[i]*d.xyz[i];
            boxfaces[nboxclip].n.xyz[i] = 1.0;
            nboxclip++;
        }
        if(ibox[1].ijk[i] > clampbox[1].ijk[i]) {
            ibox[1].ijk[i] = clampbox[1].ijk[i];
            boxfaces[nboxclip].d = clampbox[1].ijk[i]*d.xyz[i];
            boxfaces[nboxclip].n.xyz[i] = -1.0;
            nboxclip++;
        }   
    }
    if(nboxclip) kc3d_clip(poly, boxfaces, nboxclip);
}


int kc3d_clip(kc3d_poly *poly, kc3d_plane *planes, kc3d_int nplanes) {
    // direct access to vertex buffer
    kc3d_vertex *vertbuffer = poly->verts;
    kc3d_int *nverts = &poly->nverts;
    if (*nverts <= 0) return 0;

    // variable declarations
    kc3d_int v, p, np, onv, vcur, vnext, vstart, pnext, numunclipped;

    // signed distances to the clipping plane
    kc3d_real sdists[KC3D_MAX_VERTS];
    kc3d_real smin, smax;

    // for marking clipped vertices
    kc3d_int clipped[KC3D_MAX_VERTS];

    // loop over each clip plane
    for (p = 0; p < nplanes; ++p) {
        // calculate signed distances to the clip plane
        onv = *nverts;
        smin = 1.0e30;
        smax = -1.0e30;
        memset(&clipped, 0, sizeof(clipped));
        for (v = 0; v < onv; ++v) {
            sdists[v] = planes[p].d + dot(vertbuffer[v].pos, planes[p].n);
            if (sdists[v] < smin) smin = sdists[v];
            if (sdists[v] > smax) smax = sdists[v];
            if (sdists[v] < 0.0) clipped[v] = 1;
        }

        // skip this face if the poly lies entirely on one side of it
        if (smin >= 0.0) continue;
        if (smax <= 0.0) {
            *nverts = 0;
            return 1;
        }

        // check all edges and insert new vertices on the bisected edges
        for (vcur = 0; vcur < onv; ++vcur) {
            if (clipped[vcur]) continue;
            for (np = 0; np < 3; ++np) {
                vnext = vertbuffer[vcur].pnbrs[np];
                if (!clipped[vnext]) continue;
                    if (*nverts == KC3D_MAX_VERTS) {
                        #if !defined(NDEBUG)                                  
                            fprintf(stderr, "kc3d_clip: Max vertex buffer size exceeded");
                        #endif
                        return 0;
                    }
                vertbuffer[*nverts].pnbrs[0] = vcur;
                vertbuffer[vcur].pnbrs[np] = *nverts;
                wav(vertbuffer[vcur].pos, -sdists[vnext], vertbuffer[vnext].pos,
                        sdists[vcur], vertbuffer[*nverts].pos);
                (*nverts)++;
            }
        }

        // for each new vert, search around the faces for its new neighbors and
        // doubly-link everything
        for (vstart = onv; vstart < *nverts; ++vstart) {
            vcur = vstart;
            vnext = vertbuffer[vcur].pnbrs[0];
            do {
                for (np = 0; np < 3; ++np)
                    if (vertbuffer[vnext].pnbrs[np] == vcur) break;
                vcur = vnext;
                pnext = (np + 1) % 3;
                vnext = vertbuffer[vcur].pnbrs[pnext];
            } while (vcur < onv);
            vertbuffer[vstart].pnbrs[2] = vcur;
            vertbuffer[vcur].pnbrs[1] = vstart;
        }

        // go through and compress the vertex list, removing clipped verts and
        // re-indexing accordingly (reusing `clipped` to re-index everything)
        numunclipped = 0;
        for (v = 0; v < *nverts; ++v) {
            if (!clipped[v]) {
                vertbuffer[numunclipped] = vertbuffer[v];
                clipped[v] = numunclipped++;
            }
        }
        *nverts = numunclipped;
        for (v = 0; v < *nverts; ++v)
            for (np = 0; np < 3; ++np)
                vertbuffer[v].pnbrs[np] = clipped[vertbuffer[v].pnbrs[np]];
    }

    return 1;
}


#define KC3D_STACKSIZE 100


void kc3d_rasterize_local(kc3d_poly* poly, kc3d_dvec3 ibox[2], kc3d_real* dest_grid, kc3d_rvec3 d) {

    kc3d_int i, spax, dmax, nstack, siz;
    kc3d_real moments;
    kc3d_poly* children[2];
    kc3d_dvec3 gridsz;

    // return if any parameters are bad 
    for(i = 0; i < 3; ++i) gridsz.ijk[i] = ibox[1].ijk[i]-ibox[0].ijk[i];   
    if(!poly || poly->nverts <= 0 || !dest_grid || 
            gridsz.i <= 0 || gridsz.j <= 0 || gridsz.k <= 0) return;
    
    // explicit stack-based implementation
    // stack size should never overflow in this implementation, 
    // even for large input grids (up to ~512^3) 
    typedef struct {
        kc3d_poly poly;
        kc3d_dvec3 ibox[2];
    } stack_elem;

    // Small stack optimisation - only heap-allocate if needed_stacksize > KC2D_STACKSIZE
    stack_elem small_stack[KC3D_STACKSIZE];
    stack_elem *stack;

    kc3d_int needed_stacksize = (kc3d_int)(
        ceil(log2(gridsz.i))+ceil(log2(gridsz.j))+ceil(log2(gridsz.k))+1
    );

    if (KC3D_STACKSIZE < needed_stacksize)
        stack = (stack_elem*)KC3D_MALLOC(sizeof(stack_elem) * needed_stacksize);
    else
        stack = small_stack;

    // push the original polyhedron onto the stack
    // and recurse until child polyhedra occupy single voxels
    nstack = 0;
    stack[nstack].poly = *poly;
    memcpy(stack[nstack].ibox, ibox, 2*sizeof(kc3d_dvec3));
    nstack++;
    while(nstack > 0) {

        // pop the stack
        // if the leaf is empty, skip it
        --nstack;
        if(stack[nstack].poly.nverts <= 0) continue;
        
        // find the longest axis along which to split 
        dmax = 0;
        spax = 0;
        for(i = 0; i < 3; ++i) {
            siz = stack[nstack].ibox[1].ijk[i]-stack[nstack].ibox[0].ijk[i];
            if(siz > dmax) {
                dmax = siz; 
                spax = i;
            }   
        }

        // if all three axes are only one voxel long, reduce the single voxel to the dest grid
#define gind(ii, jj, kk, mm) (((ii-ibox[0].i)*gridsz.j*gridsz.k+(jj-ibox[0].j)*gridsz.k+(kk-ibox[0].k))+mm)
        if(dmax == 1) {
            kc3d_reduce(&stack[nstack].poly, &moments);
            // TODO: cell shifting for accuracy
            dest_grid[gind(stack[nstack].ibox[0].i, stack[nstack].ibox[0].j, 
                      stack[nstack].ibox[0].k, 0)] += moments;
            continue;
        }

        // split the poly and push children to the stack
        children[0] = &stack[nstack].poly;
        children[1] = &stack[nstack+1].poly;
        kc3d_split_coord(
            &stack[nstack].poly,
            children,
            d.xyz[spax]*(stack[nstack].ibox[0].ijk[spax]+dmax/2),
            spax
        );
        memcpy(stack[nstack+1].ibox, stack[nstack].ibox, 2*sizeof(kc3d_dvec3));
        stack[nstack].ibox[1].ijk[spax] -= dmax-dmax/2; 
        stack[nstack+1].ibox[0].ijk[spax] += dmax/2;
        nstack += 2;
    }
}

void kc3d_split_coord(kc3d_poly* inpoly, kc3d_poly** outpolys, kc3d_real coord, kc3d_int ax) {

    // direct access to vertex buffer
    if(inpoly->nverts <= 0) return;
    kc3d_int* nverts = &inpoly->nverts;
    kc3d_vertex* vertbuffer = inpoly->verts; 
    kc3d_int v, np, npnxt, onv, vcur, vnext, vstart, pnext, nright, cside;
    kc3d_rvec3 newpos;
    kc3d_int side[KC3D_MAX_VERTS];
    kc3d_real sdists[KC3D_MAX_VERTS];

    // calculate signed distances to the clip plane
    nright = 0;
    memset(&side, 0, sizeof(side));
    for(v = 0; v < *nverts; ++v) {
        sdists[v] = vertbuffer[v].pos.xyz[ax] - coord;
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
        for(np = 0; np < 3; ++np) {
            vnext = vertbuffer[vcur].pnbrs[np];
            if(!side[vnext]) continue;
            wav(vertbuffer[vcur].pos, -sdists[vnext],
                vertbuffer[vnext].pos, sdists[vcur],
                newpos);
            vertbuffer[*nverts].pos = newpos;
            vertbuffer[*nverts].pnbrs[0] = vcur;
            vertbuffer[vcur].pnbrs[np] = *nverts;
            (*nverts)++;
            vertbuffer[*nverts].pos = newpos;
            side[*nverts] = 1;
            vertbuffer[*nverts].pnbrs[0] = vnext;
            for(npnxt = 0; npnxt < 3; ++npnxt) 
                if(vertbuffer[vnext].pnbrs[npnxt] == vcur) break;
            vertbuffer[vnext].pnbrs[npnxt] = *nverts;
            (*nverts)++;
        }
    }

    // for each new vert, search around the faces for its new neighbors
    // and doubly-link everything
    for(vstart = onv; vstart < *nverts; ++vstart) {
        vcur = vstart;
        vnext = vertbuffer[vcur].pnbrs[0];
        do {
            for(np = 0; np < 3; ++np) if(vertbuffer[vnext].pnbrs[np] == vcur) break;
            vcur = vnext;
            pnext = (np+1)%3;
            vnext = vertbuffer[vcur].pnbrs[pnext];
        } while(vcur < onv);
        vertbuffer[vstart].pnbrs[2] = vcur;
        vertbuffer[vcur].pnbrs[1] = vstart;
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
        for(np = 0; np < 3; ++np)
            outpolys[0]->verts[v].pnbrs[np] = side[outpolys[0]->verts[v].pnbrs[np]];
    for(v = 0; v < outpolys[1]->nverts; ++v) 
        for(np = 0; np < 3; ++np)
            outpolys[1]->verts[v].pnbrs[np] = side[outpolys[1]->verts[v].pnbrs[np]];
}


void kc3d_reduce(kc3d_poly *poly, kc3d_real *moments) {

    // var declarations
    kc3d_real sixv;
    kc3d_int np;
    kc3d_int vstart, pstart, vcur, vnext, pnext;
    kc3d_rvec3 v0, v1, v2;

    // direct access to vertex buffer
    kc3d_vertex *vertbuffer = poly->verts;
    kc3d_int *nverts = &poly->nverts;

    // zero the moments
    moments[0] = 0.0;

    if (*nverts <= 0) return;

    // for keeping track of which edges have been visited
    kc3d_int emarks[KC3D_MAX_VERTS][3];
    memset(&emarks, 0, sizeof(emarks));

    // loop over all vertices to find the starting point for each face
    for (vstart = 0; vstart < *nverts; ++vstart)
        for (pstart = 0; pstart < 3; ++pstart) {
            // skip this face if we have marked it
            if (emarks[vstart][pstart]) continue;

            // initialize face looping
            pnext = pstart;
            vcur = vstart;
            emarks[vcur][pnext] = 1;
            vnext = vertbuffer[vcur].pnbrs[pnext];
            v0 = vertbuffer[vcur].pos;

            // move to the second edge
            for (np = 0; np < 3; ++np)
                if (vertbuffer[vnext].pnbrs[np] == vcur) break;
            vcur = vnext;
            pnext = (np + 1) % 3;
            emarks[vcur][pnext] = 1;
            vnext = vertbuffer[vcur].pnbrs[pnext];

            // make a triangle fan using edges and first vertex
            while (vnext != vstart) {
                v2 = vertbuffer[vcur].pos;
                v1 = vertbuffer[vnext].pos;

                sixv = (-v2.x * v1.y * v0.z + v1.x * v2.y * v0.z + v2.x * v0.y * v1.z -
                                v0.x * v2.y * v1.z - v1.x * v0.y * v2.z + v0.x * v1.y * v2.z);

                // base case
                moments[0] += ONE_SIXTH * sixv;

                // move to the next edge
                for (np = 0; np < 3; ++np)
                    if (vertbuffer[vnext].pnbrs[np] == vcur) break;
                vcur = vnext;
                pnext = (np + 1) % 3;
                emarks[vcur][pnext] = 1;
                vnext = vertbuffer[vcur].pnbrs[pnext];
            }
        }
}


void kc3d_translate(kc3d_poly *poly, kc3d_rvec3 shift) {
    kc3d_int v;
    for (v = 0; v < poly->nverts; ++v) {
        poly->verts[v].pos.x += shift.x;
        poly->verts[v].pos.y += shift.y;
        poly->verts[v].pos.z += shift.z;
    }
}
