# File   : spherical_cylinder_approx.jl
# License: GNU v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 28.08.2021


using LinearAlgebra

using GLMakie
using GeometryBasics


# Vertices and face indices generated using MeshLab; two spherical caps were created, then the
# convex hull was saved in the "spherical_cylinder.ply" file
verts = [
    2.468838e-09 -8.270091e-08 -1 
    2.468838e-09 -4.721871e-09 2 
    0.258819 0.9659259 1 
    -0.9659259 0.258819 1 
    0.9659259 -0.258819 1 
    -0.4062855 0.703707 -0.5828626 
    -0.4062854 0.703707 1.582862 
    -0.258819 -0.9659259 1 
    0.8660254 0.5 -4.371139e-08 
    0.6957635 0.2464413 1.67467 
    -0.6957635 -0.2464414 -0.6746702 
    -0.8125709 -2.265369e-09 1.582862 
    -0.7071068 0.7071068 1.000345 
    0.4062854 -0.703707 1.582862 
    0.4062855 -0.7037071 -0.5828624 
    0.258819 0.9659259 -8.444392e-08 
    0.7071068 0.7071068 1.000345 
    0.2761805 0.4783586 1.833605 
    -0.4167868 0.2406319 -0.8765758 
    -0.9659259 0.258819 -2.262668e-08 
    -0.4167868 0.240632 1.876576 
    -0.8660254 -0.5 1 
    -0.258819 0.9659259 1 
    -0.6957635 0.2464413 1.67467 
    0.4167868 -0.240632 1.876576 
    -0.4062855 -0.703707 1.582862 
    -0.2761805 -0.4783587 -0.8336051 
    0.9659259 -0.258819 2.262668e-08 
    0.258819 -0.9659259 1 
    0.5613061 0.4793281 -0.6746704 
    0.9659259 0.258819 1 
    0.5 0.8660254 1 
    0.1344573 0.7257695 1.67467 
    -0.7071068 0.7071068 -0.0003454111 
    -0.6957635 0.2464413 -0.6746703 
    -0.1385453 0.2399675 1.960844 
    -0.7071068 -0.7071068 -0.0003452875 
    -0.9659259 -0.258819 1 
    -0.5 0.8660254 1 
    0 1 1 
    -0.5613061 0.4793282 1.67467 
    -0.8660254 0.5 1 
    -0.552361 2.718866e-10 1.833605 
    0.6957635 -0.2464413 1.67467 
    0.2770906 -1.700542e-10 1.960844 
    -0.1344573 -0.7257695 1.67467 
    -0.7071068 -0.7071068 1.000345 
    -0.6957635 -0.2464413 1.67467 
    0.2761805 -0.4783587 -0.8336051 
    -0.2770906 -8.282016e-08 -0.9608439 
    0 -1 8.742278e-08 
    0.7071068 -0.7071068 1.000345 
    0.7071068 0.7071068 -0.0003454111 
    0.812571 -4.863583e-08 -0.5828623 
    0.1344573 0.7257694 -0.6746703 
    0.8660254 0.5 1 
    0.9659259 0.258819 -2.262668e-08 
    0.4062855 0.703707 1.582862 
    -0.258819 0.9659259 -8.444392e-08 
    -0.552361 -7.314797e-08 -0.8336051 
    -0.8125709 -4.869008e-08 -0.5828624 
    -0.5613062 0.4793281 -0.6746703 
    -0.8660254 0.5 -4.371139e-08 
    4.927248e-09 0.481264 1.876576 
    -0.4062854 -0.7037071 -0.5828624 
    -1 1.224647e-16 1 
    -0.9659259 -0.258819 2.262668e-08 
    -0.1344573 0.7257695 1.67467 
    -0.2770906 -1.17948e-09 1.960844 
    0.5613061 -0.4793282 1.67467 
    0.552361 -2.306788e-10 1.833605 
    0.812571 -2.31961e-09 1.582862 
    0.1385453 -0.2399675 1.960844 
    0.1385453 0.2399675 1.960844 
    0.1344573 -0.7257695 1.67467 
    -0.5613062 -0.4793282 1.67467 
    -0.5 -0.8660254 1 
    0.1385453 -0.2399676 -0.9608439 
    0.1344573 -0.7257695 -0.6746702 
    -0.1385453 0.2399674 -0.9608439 
    0 -1 1 
    0.5 -0.8660254 7.571035e-08 
    -0.258819 -0.9659259 8.444392e-08 
    0.5 -0.8660254 1 
    0.8660254 -0.5 1 
    0.5 0.8660254 -7.571035e-08 
    0.6957635 0.2464413 -0.6746702 
    0.6957635 -0.2464414 -0.6746702 
    1 0 1 
    1 0 0 
    0.5613062 0.4793282 1.67467 
    -0.5 0.8660254 -7.571035e-08 
    0 1 -8.742278e-08 
    -0.4167868 -0.2406321 -0.8765758 
    -0.2761805 0.4783585 -0.8336052 
    -0.2761805 0.4783586 1.833605 
    -0.5613061 -0.4793282 -0.6746703 
    -0.8660254 -0.5 4.371139e-08 
    -1 -1.224647e-16 1.07062e-23 
    0.4167869 0.240632 1.876576 
    -0.1385453 -0.2399675 1.960844 
    0.2761805 -0.4783586 1.833605 
    -0.4167868 -0.240632 1.876576 
    -0.2761805 -0.4783586 1.833605 
    4.927248e-09 -0.4812641 -0.8765757 
    -0.1385453 -0.2399676 -0.9608439 
    -0.1344573 -0.7257695 -0.6746702 
    0.258819 -0.9659259 8.444392e-08 
    0.2761805 0.4783586 -0.8336052 
    0.7071068 -0.7071068 -0.0003452875 
    -0.5 -0.8660254 7.571035e-08 
    0.4062854 0.703707 -0.5828626 
    0.2770906 -8.382958e-08 -0.9608439 
    0.5613062 -0.4793282 -0.6746702 
    0.552361 -7.26454e-08 -0.8336051 
    -0.1344573 0.7257694 -0.6746703 
    -4.927248e-09 -0.481264 1.876576 
    -4.927248e-09 0.4812639 -0.8765758 
    0.8660254 -0.5 4.371139e-08 
    0.1385453 0.2399674 -0.9608439 
    0.4167868 0.2406319 -0.8765758 
    0.4167869 -0.2406321 -0.8765757 
]


faces = [
    23 3 11 
    38 12 6 
    38 6 22 
    39 32 2 
    40 6 12 
    40 23 20 
    41 3 23 
    41 40 12 
    41 23 40 
    42 23 11 
    42 20 23 
    45 25 7 
    47 42 11 
    47 37 21 
    47 11 37 
    52 8 29 
    55 9 30 
    55 52 16 
    55 8 52 
    56 55 30 
    56 8 55 
    57 31 2 
    57 2 32 
    57 16 31 
    57 32 17 
    59 34 18 
    59 18 49 
    60 19 34 
    60 59 10 
    60 34 59 
    61 18 34 
    61 33 5 
    62 34 19 
    62 61 34 
    62 33 61 
    62 41 12 
    62 12 33 
    62 19 3 
    62 3 41 
    63 17 32 
    65 37 11 
    65 11 3 
    65 3 19 
    66 60 10 
    67 39 22 
    67 22 6 
    67 32 39 
    67 63 32 
    68 35 20 
    68 20 42 
    68 1 35 
    69 43 24 
    69 13 51 
    70 24 43 
    70 44 24 
    71 30 9 
    71 43 4 
    71 70 43 
    71 9 70 
    72 44 1 
    72 24 44 
    73 35 1 
    73 1 44 
    73 63 35 
    73 17 63 
    74 28 13 
    75 46 25 
    75 21 46 
    75 47 21 
    76 7 25 
    76 25 46 
    78 48 14 
    79 49 18 
    79 0 49 
    80 28 74 
    80 74 45 
    80 45 7 
    82 80 7 
    82 50 80 
    82 7 76 
    83 51 13 
    83 13 28 
    84 4 43 
    84 69 51 
    84 43 69 
    85 15 2 
    85 2 31 
    85 31 16 
    85 16 52 
    86 29 8 
    86 56 53 
    86 8 56 
    87 53 27 
    88 71 4 
    88 30 71 
    89 27 53 
    89 53 56 
    89 88 4 
    89 4 27 
    89 56 30 
    89 30 88 
    90 57 17 
    90 9 55 
    90 55 16 
    90 16 57 
    91 58 5 
    91 5 33 
    91 33 12 
    91 12 38 
    91 38 22 
    91 22 58 
    92 58 22 
    92 22 39 
    92 39 2 
    92 2 15 
    92 15 54 
    93 59 49 
    93 10 59 
    94 61 5 
    94 18 61 
    94 79 18 
    95 20 35 
    95 35 63 
    95 40 20 
    95 63 67 
    95 67 6 
    95 6 40 
    96 64 36 
    96 26 64 
    96 93 26 
    96 10 93 
    97 66 10 
    97 46 21 
    97 36 46 
    97 21 37 
    97 37 66 
    97 96 36 
    97 10 96 
    98 65 19 
    98 66 37 
    98 37 65 
    98 19 60 
    98 60 66 
    99 70 9 
    99 90 17 
    99 9 90 
    99 17 73 
    99 73 44 
    99 44 70 
    100 72 1 
    100 1 68 
    101 24 72 
    101 69 24 
    101 74 13 
    101 13 69 
    102 42 47 
    102 47 75 
    102 100 68 
    102 68 42 
    103 75 25 
    103 25 45 
    103 102 75 
    103 100 102 
    104 77 48 
    104 48 78 
    105 49 0 
    105 0 77 
    105 93 49 
    105 77 104 
    105 104 26 
    105 26 93 
    106 64 26 
    106 78 50 
    106 104 78 
    106 26 104 
    106 82 64 
    106 50 82 
    107 78 14 
    107 50 78 
    107 14 81 
    107 83 28 
    107 81 83 
    107 80 50 
    107 28 80 
    109 81 14 
    109 83 81 
    109 51 83 
    110 82 76 
    110 76 46 
    110 46 36 
    110 36 64 
    110 64 82 
    111 54 15 
    111 15 85 
    111 85 52 
    111 52 29 
    111 108 54 
    111 29 108 
    112 77 0 
    113 14 48 
    113 109 14 
    114 86 53 
    114 53 87 
    115 92 54 
    115 94 5 
    115 5 58 
    115 58 92 
    116 101 72 
    116 72 100 
    116 100 103 
    116 103 45 
    116 45 74 
    116 74 101 
    117 79 94 
    117 94 115 
    117 115 54 
    117 54 108 
    118 87 27 
    118 113 87 
    118 109 113 
    118 27 4 
    118 4 84 
    118 84 51 
    118 51 109 
    119 112 0 
    119 0 79 
    119 79 117 
    119 117 108 
    120 114 112 
    120 119 108 
    120 112 119 
    120 108 29 
    120 29 86 
    120 86 114 
    121 113 48 
    121 114 87 
    121 87 113 
    121 48 77 
    121 77 112 
    121 112 114 
]


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


# Create spherical cylinder spanning two points p1 and p2 on a particle trajectory
p1 = [5, 4, 3]
r1 = 2

p2 = [6, 7, 8]
r2 = 4


# Expand vertices to given particles' radii
verts_split[1:half, :] .*= r1
verts_split[half + 1:end, :] .*= r2


# Create rotation matrix from vertical +Z to |p2 - p1|
a = [0., 0., 1.]
b = [0., 0., 0.]

# Compute b = |p2 - p1|
s = 0.
for i in 1:3
    b[i] = p2[i] - p1[i]
    global s += b[i] * b[i]
end

# Divide by ||p2 - p1|| to get unity vector
s = sqrt(s)
for i in 1:3
    b[i] /= s
end


# Compute cross product between a and b
v = [0., 0., 0.]
v[1] = a[2] * b[3] - a[3] * b[2]
v[2] = a[3] * b[1] - a[1] * b[3]
v[3] = a[1] * b[2] - a[2] * b[1]


# Compute dot product between a and b
c = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]


# Create skew-symmetric cross-product matrix of v
vx = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
vx[2] = -v[3]
vx[3] = v[2]
vx[4] = v[3]
vx[6] = -v[1]
vx[7] = -v[2]
vx[8] = v[1]


# Multiply vx by vx
vx2 = [0., 0., 0., 0., 0., 0., 0., 0., 0.]
for i in 1:3
    for j in 1:3
        for k in 1:3
            vx2[(i - 1) * 3 + j] += vx[(i - 1) * 3 + k] * vx[(k - 1) * 3 + j]
        end
    end
end


# Create rotation matrix as I(3) + vx + vx.vx / (1 + c)
rot = [1., 0., 0., 0., 1., 0., 0., 0., 1.]

if abs(1 + c) > 1e-6
    for i in 1:9
        rot[i] += vx[i] + vx2[i] / (1 + c)
    end
end

# Rotate all vertices
temp = [0., 0., 0.]
for i in 1:size(verts_split)[1]
    for j in 1:3
        temp[j] = verts_split[i, j]
    end

    for j in 1:3
        verts_split[i, j] = rot[(j - 1) * 3 + 1] * temp[1] + 
                            rot[(j - 1) * 3 + 2] * temp[2] +
                            rot[(j - 1) * 3 + 3] * temp[3]
    end
end


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
