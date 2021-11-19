#!/usr/bin/env julia
# -*- coding: utf-8 -*-
# File   : sphero_cylinder_approx.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 21.05.2021


using GLMakie
using Polyhedra
using ColorSchemes
using LinearAlgebra

import CDDLib
lib = CDDLib.Library()


# Return points on a sphere, generated with the method of Markus Deserno from
# https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
function half_sphere(;centre = [0. 0. 0.], radius = 1., n = 64)
    a = 4π * radius^2 / 2n
    d = √a

    mθ = floor(π / d)

    dθ = π / mθ
    dϕ = a / dθ

    points = zeros(n, 3)

    # Half sphere without middle circumference
    nc = 0

    for m in 0:mθ / 2 - 2
        θ = π * (m + 0.5) / mθ
        mϕ = floor(2π * sin(θ) / dϕ)
        for p in 0:mϕ - 1
            ϕ = 2π * p / mϕ
            nc += 1

            points[nc, 1] = centre[1] + radius * sin(θ) * cos(ϕ)
            points[nc, 2] = centre[2] + radius * sin(θ) * sin(ϕ)
            points[nc, 3] = centre[3] + radius * cos(θ)
        end
    end

    # Add remaining points on the middle circumference
    inc = 2π / (n - nc)
    for i in 0:n - nc - 1
        nc += 1

        θ = π / 2
        ϕ = i * inc

        points[nc, 1] = centre[1] + radius * sin(θ) * cos(ϕ)
        points[nc, 2] = centre[2] + radius * sin(θ) * sin(ϕ)
        points[nc, 3] = centre[3] + 0.
    end

    points
end


# Construct and return the rotation matrix that rotates a unit vector a onto unit vector b, taken
# from https://math.stackexchange.com/questions/180418
function reorient(a, b)
    v = vec(a) × vec(b)
    vx = [
        0 -v[3] v[2]
        v[3] 0 -v[1]
        -v[2] v[1] 0
    ]

    I(3) + vx + vx * vx ./ (1 + a ⋅ b)
end


function sphero_cylinder(;c1 = [0. -1. -1.], c2 = [4. 4. 0.], r1 = 1., r2 = 1., n = 128)
    hs1 = half_sphere(n = n ÷ 2)
    hs2 = half_sphere(n = n ÷ 2)

    # Re-orient the half-spheres
    v1 = (c1 - c2)
    v1 ./= norm(v1)
    rot1 = reorient([0 0 1], v1)

    for i in 1:size(hs1)[1]
        hs1[i, :] .= rot1 * hs1[i, :]
    end

    v2 = -v1
    rot2 = reorient([0 0 1], v2)

    for i in 1:size(hs2)[1]
        hs2[i, :] .= rot2 * hs2[i, :]
    end

    hs1 = (hs1 .+ c1) .* r1
    hs2 = (hs2 .+ c2) .* r2

    [hs1 ; hs2]
end


# Points on a half-sphere
sphere_points = half_sphere()
sphere_poly = polyhedron(vrep(sphere_points), lib)

# Write the half-sphere vertices to a file
open("half_sphere.csv", "w") do io
    write(io, "Points on a half-sphere polyhedral approximation with 64 vertices\n")
    for i in 1:size(sphere_points)[1]
        for j in 1:size(sphere_points)[2]
            write(io, "$(sphere_points[i, j])\n")
        end
    end
end

# Points on a sphero-cylinder
sc_points = sphero_cylinder()
sc_poly = polyhedron(vrep(sc_points), lib)

# Plot the half-sphere and a sphero-cylinder
cm = ColorSchemes.magma
GLMakie.mesh(Polyhedra.Mesh{3}(sc_poly), color = get(cm, 0.5))
GLMakie.wireframe!(Polyhedra.Mesh{3}(sc_poly))

current_figure()
