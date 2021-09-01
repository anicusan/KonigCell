#!/usr/bin/env julia
# -*- coding: utf-8 -*-
# File   : circle_approx.jl
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 21.05.2021


using GLMakie
using LazySets


# Return points on a circle
function circle(;centre = [0., 0.], radius = 1., n = 16)
    points = zeros(2, n + 1)
    for (i, α) in enumerate(LinRange(0, 2pi, n + 1))
        points[1, i] = centre[1] + radius * cos(α)
        points[2, i] = centre[2] + radius * sin(α)
    end

    points
end


# Analytical circle points
t = LinRange(0, 2pi, 100)
circle_analytical = [cos.(t) sin.(t)]


# Ploting
fig = Figure()
axes = [Axis(fig[1, i]) for i in 1:2]
for ax in axes
    ax.aspect = AxisAspect(1)
end

lines!(axes[1], circle_analytical[:, 1], circle_analytical[:, 2], color = :red, linewidth = 3,
       label = "∞")


# Plot polygonal approximations and area ratio
nverts = collect(4:64)
ratios = zeros(length(nverts))
to_plot = [4, 8, 16, 32]

for (i, n) in enumerate(nverts)
    circle_points = circle(n = Int64(n))
    circle_poly = VPolygon(circle_points)
    ratios[i] = area(circle_poly) / pi

    if n in to_plot
        lines!(axes[1], circle_points[1, :], circle_points[2, :], label = "$n")
    end
end

lines!(axes[2], nverts, ratios)

axislegend(axes[1])
fig
