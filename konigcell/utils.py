#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : utils.py
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 31.08.2021


import textwrap
import numpy as np


def get_cutoffs(rmax, *args):
    '''Return physical limits encompassing each `arg` with extra `rmax` units.
    Each `arg` is horizontally stacked / concatenated.
    '''
    cutoffs = []
    for arg in args:
        arg = np.hstack(arg)
        cutoffs.append([
            np.nanmin(arg) - rmax,
            np.nanmax(arg) + rmax,
        ])

    if len(cutoffs) == 1:
        return cutoffs[0]

    return cutoffs


def check_lens(ignore_none = True, **kwargs):
    '''Raise ValueError if two elements in the input keyword arguments do not
    have the same length.
    '''
    if not len(kwargs):
        return

    # Extract first dict element's length
    k0, v0 = list(kwargs.items())[0]
    for k, v in kwargs.items():
        if ignore_none and v is None:
            continue

        if len(v) != len(v0):
            raise ValueError(textwrap.fill((
                f"All inputs must have the same length; `{k0}` has length "
                f"{len(v0)}, but `{k}` has length {len(v)}."
            )))


def check_ncols(ncols, **kwargs):
    '''Raise ValueError if any element in the input keyword arguments does not
    have the number of columns `ncols`.
    '''
    for k, v in kwargs.items():
        if v.ndim != 2 or v.shape[1] != ncols:
            raise ValueError(textwrap.fill((
                f"The input `{k}` must have exactly {ncols} columns; it had "
                f"`{k}.shape = {v.shape}`."
            )))


def split(x, n, overlap = 0):
    '''Split `x` into `n` (approx.) equal chunks with `overlap` overlapping
    elements between consecutive chunks.
    '''
    if x is None:
        return [None] * n

    s = int(np.floor(len(x) / n))
    chunks = [x[s * i:s * (i + 1) + overlap] for i in range(n - 1)]
    chunks.append(x[s * (n - 1):])

    return chunks
