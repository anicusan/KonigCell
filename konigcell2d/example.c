/**
 * File   : example.c
 * License: MIT
 * Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
 * Date   : 21.05.2021
 */


#include "konigcell2d.h"
#include <stdio.h>
#include <stdlib.h>


int             main(void)
{
    // Initialise pixel grid
    kc2d_int    dims[2] = {2048, 1080};
    kc2d_real   xlim[2] = {-10., 40.};
    kc2d_real   ylim[2] = {-10., 20.};

    kc2d_real   *grid = (kc2d_real*)calloc((size_t)dims[0] * dims[1], sizeof(kc2d_real));
    kc2d_real   *intersections = (kc2d_real*)calloc((size_t)dims[0] * dims[1], sizeof(kc2d_real));

    kc2d_pixels pixels = {
        .grid = grid,
        .dims = dims,
        .xlim = xlim,
        .ylim = ylim
    };

    // Initialise trajectory
    kc2d_int    num_particles = 5;
    kc2d_real   positions[10] = {15., 5., 35., 10., -10., 15., 5., -5., 30., -5.};
    kc2d_real   radii[5] = {1.001, 2.001, 3.001, 2.001, 2.};

    kc2d_real   *factors = (kc2d_real*)calloc((size_t)dims[0] * dims[1], sizeof(kc2d_real));

    for (kc2d_int i = 0; i < num_particles; ++i)
        factors[i] = 1;

    kc2d_particles particles = (kc2d_particles){
        .positions = positions,
        .radii = radii,
        .factors = NULL,
        .num_particles = num_particles
    };

    // Dynamic particle
    kc2d_dynamic(&pixels, &particles, kc2d_intersection, 0);

    // Rasterizing / pixellising single circular particle
    kc2d_poly    circle;
    kc2d_rvec2   centre = {{10., 0.}};
    kc2d_circle(&circle, centre, radii[0]);

    kc2d_rasterize(&circle, 1, 1, &pixels, NULL, kc2d_intersection);

    // Static particles
    for (kc2d_int i = 0; i < 2 * particles.num_particles; ++i)
        particles.positions[i] += 2.5;                      // Move particles

    kc2d_static(&pixels, &particles, kc2d_intersection);

    // Print global grid to the terminal
    printf("%f %f %f %f\n", xlim[0], xlim[1], ylim[0], ylim[1]);

    // Print pixel values
    for (kc2d_int i = 0; i < dims[0]; ++i)
    {
        for (kc2d_int j = 0; j < dims[1]; ++j)
            printf("%f ", grid[i * dims[1] + j]);
        printf("\n");
    }

    free(grid);
    free(intersections);
    free(factors);

    return 0;
}
