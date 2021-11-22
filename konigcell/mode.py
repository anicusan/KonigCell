#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : mode.py
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.05.2021


'''Cell Value Weighting Modes

Attributes
----------
RATIO : int32
    Weight cell values by the ratio of the intersected cell area / volume and
    the total circle area / sphere volume.

INTERSECTION : int32
    Weight cell values by their intersected area / volume with the rasterized
    shape.

PARTICLE : int32
    Weight cell values by the rasterized shape's area / volume.

ONE : int32
    Add factors to cells with no weighting.

'''


import numpy as np


RATIO = np.int32(0)
INTERSECTION = np.int32(1)
PARTICLE = np.int32(2)
ONE = np.int32(3)
