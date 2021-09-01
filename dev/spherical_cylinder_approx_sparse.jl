# File   : spherical_cylinder_approx.jl
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 28.08.2021


using DelimitedFiles
using LinearAlgebra

using GLMakie
using GeometryBasics


# Vertices and face indices generated using Autodesk; two spherical caps were created, then the
# convex hull was saved in the "spherical_cylinder_sparse.obj" file
verts = [
    -0.383022 0.321394 0.000000
    -0.375000 0.216506 -0.250000
    -0.500000 -0.000000 0.000000
    -0.426434 -0.075192 -0.250000
    -0.383022 -0.321394 -0.000000
    -0.278335 -0.331707 -0.250000
    -0.086824 -0.492404 -0.000000
    0.000000 -0.433013 -0.250000
    0.250000 -0.433013 -0.000000
    0.278335 -0.331707 -0.250000
    0.469846 -0.171010 0.000000
    0.426434 -0.075192 -0.250000
    0.469846 0.171010 0.000000
    0.375000 0.216506 -0.250000
    0.250000 0.433013 0.000000
    0.148099 0.406899 -0.250000
    -0.086824 0.492404 0.000000
    -0.148099 0.406899 -0.250000
    0.000000 -0.250000 -0.433013
    0.000000 0.000000 -0.500000
    0.160697 -0.191511 -0.433013
    0.246202 -0.043412 -0.433013
    0.216506 0.125000 -0.433013
    0.085505 0.234923 -0.433013
    -0.085505 0.234923 -0.433013
    -0.216506 0.125000 -0.433013
    -0.246202 -0.043412 -0.433013
    -0.160697 -0.191511 -0.433013
    -0.500000 -0.000000 1.000000
    -0.383022 0.321394 1.000000
    -0.086824 0.492404 1.000000
    0.250000 0.433013 1.000000
    0.469846 0.171010 1.000000
    0.469846 -0.171010 1.000000
    0.250000 -0.433013 1.000000
    -0.086824 -0.492404 1.000000
    -0.383022 -0.321394 1.000000
    -0.426434 -0.075192 1.250000
    -0.375000 0.216506 1.250000
    -0.148099 0.406899 1.250000
    0.148099 0.406899 1.250000
    0.375000 0.216506 1.250000
    0.426434 -0.075192 1.250000
    0.278335 -0.331707 1.250000
    -0.000000 -0.433013 1.250000
    -0.278335 -0.331707 1.250000
    -0.000000 -0.250000 1.433013
    -0.000000 -0.000000 1.500000
    -0.160697 -0.191511 1.433013
    -0.246202 -0.043412 1.433013
    -0.216506 0.125000 1.433013
    -0.085505 0.234923 1.433013
    0.085505 0.234923 1.433013
    0.216506 0.125000 1.433013
    0.246202 -0.043412 1.433013
    0.160697 -0.191511 1.433013
]


faces = [
    1 2 3
    3 2 4
    3 4 5
    5 4 6
    5 6 7
    7 6 8
    7 8 9
    9 8 10
    9 10 11
    11 10 12
    11 12 13
    13 12 14
    13 14 15
    15 14 16
    15 16 17
    17 16 18
    17 18 1
    1 18 2
    19 20 21
    21 20 22
    22 20 23
    24 20 25
    25 20 26
    27 20 28
    28 20 19
    8 19 10
    10 19 21
    10 21 12
    12 21 22
    12 22 14
    14 22 23
    14 23 16
    16 23 24
    16 24 18
    18 24 25
    18 25 2
    2 25 26
    2 26 4
    4 26 27
    4 27 6
    6 27 28
    6 28 8
    8 28 19
    23 20 24
    26 20 27
    5 29 3
    3 29 30
    3 30 1
    1 30 31
    1 31 17
    17 31 32
    17 32 15
    15 32 33
    15 33 13
    13 33 34
    13 34 11
    11 34 35
    11 35 9
    9 35 36
    9 36 7
    7 36 37
    7 37 5
    5 37 29
    37 38 29
    29 38 39
    29 39 30
    30 39 40
    30 40 31
    31 40 41
    31 41 32
    32 41 42
    32 42 33
    33 42 43
    33 43 34
    34 43 44
    34 44 35
    35 44 45
    35 45 36
    36 45 46
    36 46 37
    37 46 38
    47 48 49
    49 48 50
    50 48 51
    52 48 53
    53 48 54
    55 48 56
    56 48 47
    45 47 46
    46 47 49
    46 49 38
    38 49 50
    38 50 39
    39 50 51
    39 51 40
    40 51 52
    40 52 41
    41 52 53
    41 53 42
    42 53 54
    42 54 43
    43 54 55
    43 55 44
    44 55 56
    44 56 45
    45 56 47
    54 48 55
    51 48 52
]

# Index from 0, as expected by the C code
faces .-= 1


# Split vertices into the bottom and top caps
half = size(verts)[1] ÷ 2

verts_bottom = zeros(half, 3)
index_bottom = 1

verts_top = zeros(half, 3)
index_top = 1

faces_split = copy(faces)

for (i, v) in enumerate(eachrow(verts))
    if v[3] < 0.5
        verts_bottom[index_bottom, :] .= v
        faces_split[faces .== i - 1] .= index_bottom - 1

        global index_bottom += 1
    else
        verts_top[index_top, :] .= v
        faces_split[faces .== i - 1] .= index_top - 1 + half

        global index_top += 1
    end
end

verts_split = [verts_bottom ; verts_top]


# Move top cap back to (0, 0, 0) and flip it
verts_split[half + 1:end, 3] .-= 1
verts_split[half + 1:end, 3] .*= -1

# Expand vertices from diameter 2 to radius 2
verts_split .*= 2


# Write processed vertices and face indices to CSV files
open("spherical_cylinder_verts_sparse.csv", "w") do io
    writedlm(io, verts_split, ',')
end

open("spherical_cylinder_faces_sparse.csv", "w") do io
    writedlm(io, faces_split, ',')
end



# Create spherical cylinder spanning two points p1 and p2 on a particle trajectory
p1 = [5, 4, 3]
r1 = 2

p2 = [6, 7, 8]
r2 = 4


# Expand vertices to given particles' radii
verts_split[1:half, :] .*= r1
verts_split[half + 1:end, :] .*= r2


# Create rotation matrix from vertical +Z to |p2 - p1|
a = [0, 0, 1]
b = normalize(p2 .- p1)

v = cross(a, b)
c = dot(a, b)
vx = [  0    -v[3]   v[2]
       v[3]    0    -v[1]
      -v[2]   v[1]    0  ]

rot = I(3) + vx + vx * vx / (1 + c)

# Rotate all vertices
verts_split = (rot * verts_split')'


# Translate vertices
verts_split[1:half, :] .+= p1'
verts_split[half + 1:end, :] .+= p2'


# Plot mesh and direction from p1 to p2
m = GeometryBasics.Mesh(
    Point3.(eachrow(verts_split)),
    TriangleFace.(eachrow(faces_split .+ 1)),
)

wireframe(m)
lines!([p1 p2], color = :red)

current_figure()
