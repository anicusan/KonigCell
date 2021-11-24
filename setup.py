#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : setup.py
# License: MIT v3.0
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.05.2021


import  os
import  sys
import  warnings
import  shutil

import  setuptools


try:
    import  numpy               as      np
    from    Cython.Build        import  cythonize
    from    Cython.Distutils    import  build_ext
except ImportError as e:
    warnings.warn(e.args[0])
    warnings.warn((
        'The pept package requires Cython and Numpy to be pre-installed'
    ))
    raise ImportError((
        'Cython or Numpy not found! Please install Cython and Numpy (or run '
        '`pip install -r requirements.txt`) and try again.'
    ))



name = "konigcell"
author = "Andrei Leonard Nicusan"
author_email = "a.l.nicusan@bham.ac.uk"

keywords = "grid pixel voxel field euler projection"
description = (
    "Quantitative, Fast Grid-Based Fields Calculations in 2D and 3D - "
    "Residence Time Distributions, Velocity Grids, Eulerian Cell Projections "
    "etc."
)
url = "https://github.com/anicusan/KonigCell"


with open("README.md", "r") as f:
    long_description = f.read()


def requirements(filename):
    # The dependencies are the same as the contents of requirements.txt
    with open(filename) as f:
        return [line.strip() for line in f if line.strip()]


# What packages are required for this module to be executed?
required = requirements('requirements.txt')

# What packages are optional?
extras = dict(docs = requirements('requirements_extra.txt'))

# Load the package's __version__.py module as a dictionary.
here = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(here, "konigcell", "__version__.py")) as f:
    exec(f.read(), about)


# Cythonize keyword arguments
cythonize_kw = dict(language_level = 3)

# Compiler arguments for each extension - with *full* optimisations.
# Unix-specific compiler args are followed by MSVC ones; they will be filtered
# based on the compiler used in `BuildExtCompilerSpecific`
cy_extension_kw = dict()
cy_extension_kw['extra_compile_args'] = [
    '-O3', '-flto', '/O2', '/GL'
]
cy_extension_kw['extra_link_args'] = ['-flto', '/LTCG']
cy_extension_kw['include_dirs'] = [np.get_include()]

cy_extensions = [
    setuptools.Extension(
        'konigcell.kc2d',
        ['konigcell/kc2d.pyx'],
        **cy_extension_kw
    ),
    setuptools.Extension(
        'konigcell.kc3d',
        ['konigcell/kc3d.pyx'],
        **cy_extension_kw
    ),
]

extensions = cythonize(cy_extensions, **cythonize_kw)


class BuildExtCompilerSpecific(build_ext):
    '''Before compiling extensions, ensure only valid compiler arguments are
    used - e.g. MSVC expects "/O2", while GCC and Clang expect "-O3".
    '''
    def build_extensions(self):
        # If compiling under MSVC, only allow "/*" compiler arguments
        if "msvc" in self.compiler.compiler_type.lower():
            for ext in self.extensions:
                ext.extra_compile_args = [
                    ca for ca in ext.extra_compile_args if ca.startswith("/")
                ]

                ext.extra_link_args = [
                    la for la in ext.extra_link_args if la.startswith("/")
                ]

        # Otherwise only allow compiler arguments starting with "-"
        else:
            for ext in self.extensions:
                ext.extra_compile_args = [
                    ca for ca in ext.extra_compile_args if ca.startswith("-")
                ]

                ext.extra_link_args = [
                    la for la in ext.extra_link_args if la.startswith("-")
                ]

        build_ext.build_extensions(self)


class UploadCommand(setuptools.Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            shutil.rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(f'{sys.executable} setup.py sdist bdist_wheel --universal')

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


setuptools.setup(
    name = name,
    version = about["__version__"],
    author = author,
    author_email = author_email,

    keywords = keywords,
    description = description,
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = url,

    install_requires = required,
    extras_require = extras,
    include_package_data = True,
    packages = setuptools.find_packages(),

    license = "MIT",
    classifiers = [
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Programming Language :: Python :: 3",
        "Programming Language :: Cython",
        "Programming Language :: C",
    ],

    python_requires = '>=3.6',
    # $ setup.py publish support.
    cmdclass = {
        'upload': UploadCommand,
        'build_ext': BuildExtCompilerSpecific,
    },
    ext_modules = extensions,
)
