#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : __init__.py
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.05.2021


from    .pixels         import  Pixels
from    .voxels         import  Voxels

from    .               import  kc2d


# Import package version
from    .__version__    import  __version__


# Mode enum: kc2d_mode and kc3d_mode
from    .               import  mode
from    .mode           import  RATIO, INTERSECTION, PARTICLE, ONE



__all__ = [
    'Pixels',
    'Voxels',
    'kc2d',
    'RATIO',
    'INTERSECTION',
    'PARTICLE',
    'ONE',
]


__author__ = "Andrei Leonard Nicusan"
__email__ = "a.l.nicusan@bham.ac.uk"
__license__ = "MIT"
__status__ = "Alpha"
