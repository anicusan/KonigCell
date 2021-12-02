

![KonigCell](/docs/source/_static/logo.png?raw=true "KonigCell Logo.")

[![PyPI version shields.io](https://img.shields.io/pypi/v/konigcell.svg?style=flat-square)](https://pypi.python.org/pypi/konigcell/)
[![Documentation Status](https://readthedocs.org/projects/konigcell/badge/?version=latest&style=flat-square)](https://konigcell.readthedocs.io/en/latest/?badge=latest)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/anicusan/KonigCell.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/anicusan/KonigCell/context:python)
[![Language grade: C/C++](https://img.shields.io/lgtm/grade/cpp/g/anicusan/KonigCell.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/anicusan/KonigCell/context:cpp)
[![License: MIT](https://img.shields.io/github/license/anicusan/konigcell?style=flat-square)](https://github.com/anicusan/konigcell)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/konigcell.svg?style=flat-square)](https://pypi.python.org/pypi/konigcell/)



#### **Quantitative, Fast Grid-Based Fields Calculations in 2D and 3D** - Residence Time Distributions, Velocity Grids, Eulerian Cell Projections etc.

That sounds dry as heck.

#### **Project moving particles' trajectories (experimental or simulated) onto 2D or 3D grids with infinite resolution.**

Better? No? Here are some figures produced by KonigCell:


![Example Plots](/docs/source/_static/examples.png?raw=true "Example plots produced by KonigCell.")

*Left panel: 2D residence time distribution in a GranuTools GranuDrum imaged using Positron Emission Particle Tracking (PEPT). Two middle panels: 3D velocity distribution in the same system; voxels are rendered as a scatter plot (left) and tomogram-like 3-slice (right). Right panel: velocity vectorfield in a constricted pipe simulating a aneurysm, imaged using PEPT.*


This is, to my knowledge, the only library that accurately projects particle
trajectories onto grids - that is, taking their full projected area / volume into
account (and not approximating them as points / lines). It's also the only one creating
quantitative 3D projections.

And it is *fast* - 1,000,000 particle positions can be rasterized onto a 512x512
grid in 7 seconds on my 16-thread i9 CPU. The code is fully parallelised on
threads, processes or distributed MPI nodes.



## The KonigCell Libraries

This repository effectively hosts three libraries:

- `konigcell2d`: a portable C library for 2D grid projections.
- `konigcell3d`: a portable C library for 3D grid projections.
- `konigcell`: a user-friendly Python interface to the two libraries above.



### Installing the Python Package

This package supports Python 3.6 and above (though it might work with even older
versions).

Install this package from PyPI:

```pip install konigcell``` 


Or conda-forge:

```conda install konigcell```


If you have a relatively standard system, the above should just download pre-compiled wheels -
so no prior configuration should be needed.


To *build* this package on your specific machine, you will need a C compiler -
the low-level C code does not use any tomfoolery, so any compiler since the
2000s should do.


To build the latest development version from GitHub:

```pip install git+https://github.com/anicusan/KonigCell```



### Integrating the C Libraries with your Code

The C libraries in the `konigcell2d` and `konigcell3d` directories in this repository; they
contain instructions for compiling and using the low-level subroutines. All code is fully
commented and follows a portable subset of the C99 standard - so no VLAs, weird macros or
compiler-specific extensions. Even MSVC compiles it!

You can run `make` in the `konigcell2d` or `konigcell3d` directories to build shared
libraries and the example executables under `-Wall -Werror -Wextra` like a stickler. Running
`make` in the repository root builds both libraries.

Both libraries are effectively single-source - they should be as straightforward as possible
to integrate in other C / C++ codebases, or interface with from higher-level programming
languages.



## Examples and Documentation

The `examples` directory contains some Python scripts using the high-level Python routines
and the low-level Cython interfaces. The `konigcell2d` and `konigcell3d` contain C examples.

Full documentation is available [here](https://konigcell.readthedocs.io/).

[TODO: add example code on the repo README too]



## Contributing
You are more than welcome to contribute to this library in the form of library
improvements, documentation or helpful examples; please submit them either as:

- GitHub issues.
- Pull requests (superheroes only).
- Email me at <a.l.nicusan@bham.ac.uk>.



## Acknowledgements
I would like to thank the Formulation Engineering CDT @School of Chemical
Engineering and the Positron Imaging Centre @School of Physics and
Astronomy, University of Birmingham for supporting my work.

And thanks to Dr. Kit Windows-Yule for putting up with my bonkers ideas.



## Citing
If you use this library in your research, you are kindly asked to cite:

> [Paper after publication]


This library would not have been possible without the excellent `r3d` library
(https://github.com/devonmpowell/r3d) which forms the very core of the C
subroutines; if you use KonigCell in your work, please also cite:

> Powell D, Abel T. An exact general remeshing scheme applied to physically conservative voxelization. Journal of Computational Physics. 2015 Sep 15;297:340-56.



## Licensing
KonigCell is MIT licensed. Enjoy.
