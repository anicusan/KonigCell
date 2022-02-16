#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pixels.py
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.05.2021


import  pickle
import  textwrap

import  numpy                       as      np

# Plotting is optional
try:
    import  plotly.graph_objects    as      go
except ImportError:
    pass

try:
    import  matplotlib.pyplot       as      plt
except ImportError:
    pass


class Pixels:
    '''A class managing a 2D pixel space with physical dimensions, including
    tools for pixel manipulation and visualisation.

    The `.pixels` attribute is simply a `numpy.ndarray[ndim=2, dtype=float64]`.
    If you think of `Pixels` as an image, the origin is the top left corner,
    the X-dimension is the left edge and the Y-dimension is the top edge, so
    that it can be indexed as `.pixels[ix, iy]`.

    The `.attrs` dictionary can be used to store extra information.

    Attributes
    ----------
    pixels: (M, N) np.ndarray[ndim=2, dtype=float64]
        The 2D numpy array containing the pixel values. This class assumes a
        uniform grid of pixels - that is, the pixel size in each dimension is
        constant, but can vary from one dimension to another.

    xlim: (2,) np.ndarray[ndim=1, dtype=float64]
        The lower and upper boundaries of the pixellised volume in the
        x-dimension, formatted as [x_min, x_max].

    ylim: (2,) np.ndarray[ndim=1, dtype=float64]
        The lower and upper boundaries of the pixellised volume in the
        y-dimension, formatted as [y_min, y_max].

    pixel_size: (2,) np.ndarray[ndim=1, dtype=float64]
        The lengths of a pixel in the x- and y-dimensions, respectively.

    pixel_grids: list[(M+1,) np.ndarray, (N+1,) np.ndarray]
        A list containing the pixel gridlines in the x- and y-dimensions.
        Each dimension's gridlines are stored as a numpy of the pixel
        delimitations, such that it has length (M + 1), where M is the number
        of pixels in a given dimension.

    lower: (2,) np.ndarray[ndim=1, dtype=float64]
        The lower left corner of the pixel rectangle; corresponds to
        [xlim[0], ylim[0]].

    upper: (2,) np.ndarray[ndim=1, dtype=float64]
        The upper right corner of the pixel rectangle; corresponds to
        [xlim[1], ylim[1]].

    attrs: dict[Any, Any]
        A dictionary storing any other user-defined information.

    Notes
    -----
    The class saves `pixels` as a **contiguous** numpy array for efficient
    access in C / Cython functions. The inner data can be mutated, but do not
    change the shape of the array after instantiating the class.

    Examples
    --------
    Create a zeroed 4x4 Pixels grid:

    >>> import konigcell as kc
    >>> pixels = kc.Pixels.zeros((4, 4), xlim = [0, 10], ylim = [0, 5])
    >>> pixels
    Pixels
    ------
    xlim = [ 0. 10.]
    ylim = [0. 5.]
    pixels =
      (shape: (4, 4))
      [[0. 0. 0. 0.]
       [0. 0. 0. 0.]
       [0. 0. 0. 0.]
       [0. 0. 0. 0.]]
    attrs = {}

    Or create a Pixels instance from another array (e.g. an image or matrix):

    >>> import numpy as np
    >>> matrix = np.ones((3, 3))
    >>> pixels = kc.Pixels(matrix, xlim = [0, 10], ylim = [-5, 5])
    >>> pixels
    Pixels
    ------
    xlim = [ 0. 10.]
    ylim = [-5.  5.]
    pixels =
      (shape: (3, 3))
      [[1. 1. 1.]
       [1. 1. 1.]
       [1. 1. 1.]]
    attrs = {}

    Access pixels' properties directly:

    >>> pixels.xlim             # ndarray[xmin, xmax]
    >>> pixels.ylim             # ndarray[ymin, ymax]
    >>> pixels.pixel_size       # ndarray[xsize, ysize]
    >>> pixels.pixels.shape     # pixels resolution - tuple[nx, ny]

    You can save extra attributes about the pixels instance in the `attrs`
    dictionary:

    >>> pixels.attrs["dpi"] = 300
    >>> pixels
    Pixels
    ------
    xlim = [ 0. 10.]
    ylim = [-5.  5.]
    pixels =
      (shape: (3, 3))
      [[1. 1. 1.]
       [1. 1. 1.]
       [1. 1. 1.]]
    attrs = {
      'dpi': 300
    }

    The lower left and upper right corners of the pixel grid, in physical
    coordinates (the ones given by xlim and ylim):

    >>> pixels.lower
    array([ 0., -5.])

    >>> pixels.upper
    array([10.,  5.])

    You can access the underlying NumPy array directly:

    >>> pixels.pixels
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])

    Indexing is forwarded to the NumPy array:

    >>> pixels[:, :]
    array([[1., 1., 1.],
           [1., 1., 1.],
           [1., 1., 1.]])

    Transform physical units into pixel indices:

    >>> pixels.from_physical([5, 0])                    # pixel centres
    array([1., 1.])

    >>> pixels.from_physical([5, 0], corner = True)     # lower left corners
    array([1.5, 1.5])

    Transform pixel indices into physical units:

    >>> pixels.to_physical([0, 0])                      # pixels centres
    array([ 1.66666667, -3.33333333])

    >>> pixels.to_physical([0, 0], corner = True)       # lower left corners
    array([ 0., -5.])

    Save Pixels instance to disk, as a binary archive:

    >>> pixels.save("pixels.pickle")
    >>> pixels = kc.load("pixels.pickle")

    Create deep copy of a Pixels instance:

    >>> Pixels.copy()

    Matplotlib plotting (optional, if Matplotlib is installed):

    >>> fig, ax = pixels.plot()
    >>> fig.show()

    Plotly trace (optional, if Plotly is installed):

    >>> import plotly.graph_objs as go
    >>> fig = go.Figure()
    >>> fig.add_trace(pixels.heatmap_trace())
    >>> fig.show()

    See Also
    --------
    konigcell.Voxels : a class managing a physical 3D voxel space.
    konigcell.dynamic2d : rasterize moving particles' trajectories.
    konigcell.static2d : rasterize static particles' positions.
    konigcell.dynamic_prob2d : 2D probability distribution of a quantity.
    '''
    __slots__ = ("_pixels", "_xlim", "_ylim", "_attrs", "_pixel_size",
                 "_pixel_grids", "_lower", "_upper")

    def __init__(self, pixels_array, xlim, ylim, **kwargs):
        '''`Pixels` class constructor.

        Parameters
        ----------
        pixels_array: 3D numpy.ndarray
            A 3D numpy array, corresponding to a pre-defined pixel space.

        xlim: (2,) numpy.ndarray
            The lower and upper boundaries of the pixellised volume in the
            x-dimension, formatted as [x_min, x_max].

        ylim: (2,) numpy.ndarray
            The lower and upper boundaries of the pixellised volume in the
            y-dimension, formatted as [y_min, y_max].

        kwargs: extra keyword arguments
            Extra user-defined attributes to be saved in `.attrs`.

        Raises
        ------
        ValueError
            If `pixels_array` does not have exactly 2 dimensions or if
            `xlim` or `ylim` do not have exactly 2 values each.

        Notes
        -----
        No copies are made if `pixels_array`, `xlim` and `ylim` are contiguous
        NumPy arrays with dtype=float64.

        '''

        # Type-checking inputs
        pixels_array = np.asarray(
            pixels_array,
            order = "C",
            dtype = np.float64,
        )

        if pixels_array.ndim != 2:
            raise ValueError(textwrap.fill((
                "The input `pixels_array` must contain an array-like with "
                "exactly three dimensions (i.e. pre-made pixels array). "
                f"Received an array with {pixels_array.ndim} dimensions."
            )))

        xlim = np.asarray(xlim, dtype = np.float64)

        if xlim.ndim != 1 or len(xlim) != 2 or xlim[0] >= xlim[1]:
            raise ValueError(textwrap.fill((
                "The input `xlim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the pixel space in the x-dimension. "
                f"Received parameter with shape {xlim.shape}."
            )))

        ylim = np.asarray(ylim, dtype = np.float64)

        if ylim.ndim != 1 or len(ylim) != 2 or ylim[0] >= ylim[1]:
            raise ValueError(textwrap.fill((
                "The input `ylim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the pixel space in the y-dimension. "
                f"Received parameter with shape {ylim.shape}."
            )))

        # Setting class attributes
        self._pixels = pixels_array
        self._xlim = xlim
        self._ylim = ylim
        self._attrs = dict(kwargs)


    @property
    def pixels(self):
        return self._pixels


    @property
    def xlim(self):
        return self._xlim


    @property
    def ylim(self):
        return self._ylim


    @property
    def attrs(self):
        return self._attrs


    @property
    def pixel_size(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_pixel_size"):
            self._pixel_size = np.array([
                (self._xlim[1] - self._xlim[0]) / self._pixels.shape[0],
                (self._ylim[1] - self._ylim[0]) / self._pixels.shape[1],
            ])

        return self._pixel_size


    @property
    def pixel_grids(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_pixel_grids"):
            self._pixel_grids = [
                np.linspace(lim[0], lim[1], self._pixels.shape[i] + 1)
                for i, lim in enumerate((self._xlim, self._ylim))
            ]

        return self._pixel_grids


    @property
    def lower(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_lower"):
            self._lower = np.array([self._xlim[0], self._ylim[0]])

        return self._lower


    @property
    def upper(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_upper"):
            self._upper = np.array([self._xlim[1], self._ylim[1]])

        return self._upper


    @staticmethod
    def zeros(shape, xlim, ylim, **kwargs):
        return Pixels(np.zeros(shape, dtype = float), xlim, ylim, **kwargs)


    def save(self, filepath):
        '''Save a `Pixels` instance as a binary `pickle` object.

        Saves the full object state, including the inner `.pixels` NumPy array,
        `xlim`, etc. in a fast, portable binary format. Load back the object
        using the `load` method.

        Parameters
        ----------
        filepath : filename or file handle
            If filepath is a path (rather than file handle), it is relative
            to where python is called.

        Examples
        --------
        Save a `Pixels` instance, then load it back:

        >>> import numpy as np
        >>> import konigcell as kc
        >>>
        >>> grid = np.zeros((640, 480))
        >>> pixels = kc.Pixels(grid, [0, 20], [0, 10])
        >>> pixels.save("pixels.pickle")

        >>> pixels_reloaded = kc.Pixels.load("pixels.pickle")

        '''
        with open(filepath, "wb") as f:
            pickle.dump(self, f)


    @staticmethod
    def load(filepath):
        '''Load a saved / pickled `Pixels` object from `filepath`.

        Most often the full object state was saved using the `.save` method.

        Parameters
        ----------
        filepath : filename or file handle
            If filepath is a path (rather than file handle), it is relative
            to where python is called.

        Returns
        -------
        pept.Pixels
            The loaded `pept.Pixels` instance.

        Examples
        --------
        Save a `Pixels` instance, then load it back:

        >>> import numpy as np
        >>> import konigcell as kc
        >>>
        >>> grid = np.zeros((640, 480))
        >>> pixels = kc.Pixels(grid, [0, 20], [0, 10])
        >>> pixels.save("pixels.pickle")

        >>> pixels_reloaded = kc.Pixels.load("pixels.pickle")

        '''
        with open(filepath, "rb") as f:
            obj = pickle.load(f)

        return obj


    def copy(self, pixels_array = None, xlim = None, ylim = None, **kwargs):
        '''Create a copy of the current `Pixels` instance, optionally with new
        `pixels_array`, `xlim` and / or `ylim`.

        The extra attributes in `.attrs` are propagated too. Pass new
        attributes as extra keyword arguments.
        '''
        if pixels_array is None:
            pixels_array = self.pixels.copy()
        if xlim is None:
            xlim = self.xlim.copy()
        if ylim is None:
            ylim = self.ylim.copy()

        # Propagate attributes
        kwargs.update(self.attrs)

        return Pixels(pixels_array, xlim, ylim, **kwargs)


    def from_physical(self, locations, corner = False):
        '''Transform `locations` from physical dimensions to pixel indices. If
        `corner = True`, return the index of the bottom left corner of each
        pixel; otherwise, use the pixel centres.

        Examples
        --------
        Create a simple `konigcell.Pixels` grid, spanning [-5, 5] mm in the
        X-dimension and [10, 20] mm in the Y-dimension:

        >>> import konigcell as kc
        >>> pixels = kc.Pixels.zeros((5, 5), xlim=[-5, 5], ylim=[10, 20])
        >>> pixels
        Pixels
        ------
        xlim = [-5.  5.]
        ylim = [10. 20.]
        pixels =
          (shape: (5, 5))
          [[0. 0. ... 0. 0.]
           [0. 0. ... 0. 0.]
           ...
           [0. 0. ... 0. 0.]
           [0. 0. ... 0. 0.]]
        attrs = {}

        >>> pixels.pixel_size
        array([2., 2.])

        Transform physical coordinates to pixel coordinates:

        >>> pixels.from_physical([-5, 10], corner = True)
        array([0., 0.])

        >>> pixels.from_physical([-5, 10])
        array([-0.5, -0.5])

        The pixel coordinates are returned exactly, as real numbers. For pixel
        indices, round them into values:

        >>> pixels.from_physical([0, 15]).astype(int)
        array([2, 2])

        Multiple coordinates can be given as a 2D array / list of lists:

        >>> pixels.from_physical([[0, 15], [5, 20]])
        array([[2. , 2. ],
               [4.5, 4.5]])

        '''

        offset = 0. if corner else self.pixel_size / 2
        return (locations - self.lower - offset) / self.pixel_size


    def to_physical(self, indices, corner = False):
        '''Transform `indices` from pixel indices to physical dimensions. If
        `corner = True`, return the coordinates of the bottom left corner of
        each pixel; otherwise, use the pixel centres.

        Examples
        --------
        Create a simple `konigcell.Pixels` grid, spanning [-5, 5] mm in the
        X-dimension and [10, 20] mm in the Y-dimension:

        >>> import konigcell as kc
        >>> pixels = kc.Pixels.zeros((5, 5), xlim=[-5, 5], ylim=[10, 20])
        >>> pixels
        Pixels
        ------
        xlim = [-5.  5.]
        ylim = [10. 20.]
        pixels =
          (shape: (5, 5))
          [[0. 0. ... 0. 0.]
           [0. 0. ... 0. 0.]
           ...
           [0. 0. ... 0. 0.]
           [0. 0. ... 0. 0.]]
        attrs = {}

        >>> pixels.pixel_size
        array([2., 2.])

        Transform physical coordinates to pixel coordinates:

        >>> pixels.to_physical([0, 0], corner = True)
        array([-5., 10.])

        >>> pixels.to_physical([0, 0])
        array([-4., 11.])

        Multiple coordinates can be given as a 2D array / list of lists:

        >>> pixels.to_physical([[0, 0], [4, 3]])
        array([[-4., 11.],
               [ 4., 17.]])

        '''

        offset = 0. if corner else self.pixel_size / 2
        return self.lower + indices * self.pixel_size + offset


    def heatmap_trace(
        self,
        colorscale = "Magma",
        transpose = True,
        xgap = 0.,
        ygap = 0.,
    ):
        '''Create and return a Plotly `Heatmap` trace of the pixels.

        Parameters
        ----------
        colorscale : str, default "Magma"
            The Plotly scheme for color-coding the pixel values in the input
            data. Typical ones include "Cividis", "Viridis" and "Magma".
            A full list is given at `plotly.com/python/builtin-colorscales/`.
            Only has an effect if `colorbar = True` and `color` is not set.

        transpose : bool, default True
            Transpose the heatmap (i.e. flip it across its diagonal).

        Examples
        --------
        Create a Pixels array and plot it as a heatmap using Plotly:

        >>> import konigcell as kc
        >>> import numpy as np
        >>> import plotly.graph_objs as go

        >>> pixels_raw = np.arange(150).reshape(10, 15)
        >>> pixels = kc.Pixels(pixels_raw, [-5, 5], [-5, 10])

        >>> fig = go.Figure()
        >>> fig.add_trace(pixels.heatmap_trace())
        >>> fig.show()

        '''

        # Compute the pixel centres
        x = self.pixel_grids[0]
        x = (x[1:] + x[:-1]) / 2

        y = self.pixel_grids[1]
        y = (y[1:] + y[:-1]) / 2

        heatmap = dict(
            x = x,
            y = y,
            z = self.pixels,
            colorscale = colorscale,
            transpose = transpose,
            xgap = xgap,
            ygap = ygap,
        )

        # If you see this error, it means you don't have Plotly; install it
        # with `pip install plotly`
        return go.Heatmap(heatmap)


    def plot(self, ax = None):
        '''Plot pixels as a heatmap using Matplotlib.

        Returns matplotlib figure and axes objects containing the pixel values
        colour-coded in a Matplotlib image (i.e. heatmap).

        Parameters
        ----------
        ax : mpl_toolkits.mplot3D.Axes3D object, optional
            The 3D matplotlib-based axis for plotting. If undefined, new
            Matplotlib figure and axis objects are created.

        Returns
        -------
        fig, ax : matplotlib figure and axes objects

        Examples
        --------
        Pixellise an array of lines and plot them with Matplotlib:

        >>> lines = np.array(...)                   # shape (N, M >= 7)
        >>> lines2d = lines[:, [0, 1, 2, 4, 5]]     # select x, y of lines
        >>> number_of_pixels = [10, 10]
        >>> pixels = pept.Pixels.from_lines(lines2d, number_of_pixels)

        >>> fig, ax = pixels.plot()
        >>> fig.show()

        '''

        # If you see this error, it means you don't have Matplotlib; install it
        # with `pip install matplotlib`
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = plt.gcf()

        # Plot the values in pixels
        ax.imshow(np.rot90(self.pixels))

        # Compute the pixel centres and set them in the Matplotlib image
        # x = self.pixel_grids[0]
        # x = (x[1:] + x[:-1]) / 2

        # y = self.pixel_grids[1]
        # y = (y[1:] + y[:-1]) / 2

        # Matplotlib shows numbers in a long format ("102.000032411"), so round
        # them to two decimals before plotting
        # ax.set_xticklabels(np.round(x, 2))
        # ax.set_yticklabels(np.round(y, 2))

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        return fig, ax


    def __getitem__(self, *args, **kwargs):
        return self.pixels.__getitem__(*args, **kwargs)


    def __repr__(self):
        # String representation of the class
        name = "Pixels"
        underline = "-" * len(name)

        # Custom printing of the .lines and .samples_indices arrays
        with np.printoptions(threshold = 5, edgeitems = 2):
            pixels_str = f"{textwrap.indent(str(self.pixels), '  ')}"

        # Pretty-printing extra attributes
        attrs_str = ""
        if self.attrs:
            items = []
            for k, v in self.attrs.items():
                s = f"  {k.__repr__()}: {v}"
                if len(s) > 75:
                    s = s[:72] + "..."
                items.append(s)
            attrs_str = "\n" + "\n".join(items) + "\n"

        # Return constructed string
        return (
            f"{name}\n{underline}\n"
            f"xlim = {self.xlim}\n"
            f"ylim = {self.ylim}\n"
            f"pixels = \n"
            f"  (shape: {self.pixels.shape})\n"
            f"{pixels_str}\n"
            "attrs = {"
            f"{attrs_str}"
            "}\n"
        )
