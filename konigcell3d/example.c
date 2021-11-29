/**
 * File   : example.c
 * License: MIT
 * Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
 * Date   : 21.05.2021
 */


#include "konigcell3d.h"
#include <stdio.h>
#include <stdlib.h>


int             main(void)
{
    // Initialise voxel grid
    kc3d_int    dims[3] = {100, 100, 100};
    kc3d_real   xlim[2] = {-10., 20.};
    kc3d_real   ylim[2] = {-10., 20.};
    kc3d_real   zlim[2] = {-10., 20.};

    kc3d_real   *grid = (kc3d_real*)calloc((size_t)dims[0] * dims[1] * dims[2], sizeof(kc3d_real));

    kc3d_voxels voxels = {
        .grid = grid,
        .dims = dims,
        .xlim = xlim,
        .ylim = ylim,
        .zlim = zlim,
    };

    // Initialise trajectory
    kc3d_int    num_particles = 4;
    kc3d_real   positions[12] = {15., 5., 5., 10., -10., -5., 5., -5., 0., -5., 15., 15.};
    kc3d_real   radii[4] = {1.001, 3.001, 2.001, 2.001};

    kc3d_particles particles = (kc3d_particles){
        .positions = positions,
        .radii = radii,
        .factors = NULL,
        .num_particles = num_particles
    };

    // Dynamic particle
    kc3d_dynamic(&voxels, &particles, kc3d_intersection, 0);

    // Rasterizing / pixellising single circular particle
    kc3d_poly    sphere;
    kc3d_rvec3   centre = {{10., 0., -5.}};
    kc3d_sphere(&sphere, centre, radii[0]);

    kc3d_rasterize(&sphere, 1, 1, &voxels, NULL, kc3d_intersection);

    // Static particles
    for (kc3d_int i = 0; i < 2 * particles.num_particles; ++i)
        particles.positions[i] += 2.5;                      // Move particles

    kc3d_static(&voxels, &particles, kc3d_intersection);

    // Print global grid to the terminal
    printf("%d %d %d\n", dims[0], dims[1], dims[2]);
    printf("%f %f %f %f %f %f\n", xlim[0], xlim[1], ylim[0], ylim[1], zlim[0], zlim[1]);

    // Print pixel values
    for (kc3d_int i = 0; i < dims[0] * dims[1] * dims[2]; ++i)
        printf("%f ", grid[i]);
    printf("\n");

    free(grid);
    return 0;
}
