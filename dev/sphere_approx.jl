#!/usr/bin/env julia
# -*- coding: utf-8 -*-
# File   : sphere_approx.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 21.05.2021


using GLMakie
using Polyhedra
using ColorSchemes
using LinearAlgebra

import QHull
lib = QHull.Library()


# Return points on a sphere, generated with the method of Markus Deserno from
# https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
function sphere(;centre = [0., 0., 0.], radius = 1., n = 16)
    nc = 0
    a = 4π * radius^2 / n
    d = √a

    mθ = floor(π / d)

    dθ = π / mθ
    dϕ = a / dθ

    points = zeros(n, 3)

    for m in 0:mθ - 1
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

    return points[1:nc, :]
end


# Very dense circle points
sphere_points = sphere(n = 512)
sphere_poly = polyhedron(vrep(sphere_points), lib)


# Plot polygonal approximations and area ratio
cm = ColorSchemes.magma
# GLMakie.mesh(Polyhedra.Mesh{3}(sphere_poly), color = get(cm, 0.))

nverts = collect(10:25:500)   # [16, 32, 64, 128, 256]
ratios = zeros(length(nverts))
colors = [get(cm, i) for i in LinRange(0.1, 0.9, length(nverts))]

for (i, n) in enumerate(nverts)
    local sphere_points = sphere(centre = [0., 0. - i / 10, 0.], n = n)

    local sphere_poly = polyhedron(vrep(sphere_points), lib)
    # GLMakie.mesh!(Polyhedra.Mesh{3}(sphere_poly), color = colors[i])

    ratios[i] = Polyhedra.volume(sphere_poly) / (4 / 3 * pi)
end


current_figure()


# Ploting
GLMakie.lines(nverts, ratios)
