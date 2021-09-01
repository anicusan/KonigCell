#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : pixels.py
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.05.2021


import  pickle
import  textwrap

import  numpy                   as      np

import  plotly.graph_objects    as      go
import  matplotlib.pyplot       as      plt


class Pixels(np.ndarray):
    '''A `numpy.ndarray` subclass that manages a 2D pixel space, including
    tools for pixel manipulation and visualisation.

    Besides a 2D array of pixels, this class also stores the physical space
    spanned by the pixel grid `xlim` and `ylim` along with some pixel-specific
    helper methods.

    This subclasses the `numpy.ndarray` class, so any `Pixels` object acts
    exactly like a 2D numpy array. All numpy methods and operations are valid
    on `Pixels` (e.g. add 1 to all pixels with `pixels += 1`).

    Attributes
    ----------
    pixels: (M, N) numpy.ndarray
        The 2D numpy array containing the number of lines that pass through
        each pixel. They are stored as `float`s. This class assumes a uniform
        grid of pixels - that is, the pixel size in each dimension is constant,
        but can vary from one dimension to another. The number of pixels in
        each dimension is defined by `number_of_pixels`.

    xlim: (2,) numpy.ndarray
        The lower and upper boundaries of the pixellised volume in the
        x-dimension, formatted as [x_min, x_max].

    ylim: (2,) numpy.ndarray
        The lower and upper boundaries of the pixellised volume in the
        y-dimension, formatted as [y_min, y_max].

    pixel_size: (2,) numpy.ndarray
        The lengths of a pixel in the x- and y-dimensions, respectively.

    pixel_grid: list[numpy.ndarray]
        A list containing the pixel gridlines in the x- and y-dimensions.
        Each dimension's gridlines are stored as a numpy of the pixel
        delimitations, such that it has length (M + 1), where M is the number
        of pixels in a given dimension.

    pixel_lower: numpy.ndarray
        The lower left corner of the pixel rectangle.

    pixel_upper: numpy.ndarray
        The upper right corner of the pixel rectangle.

    Notes
    -----
    The class saves `pixels` as a **contiguous** numpy array for efficient
    access in C / Cython functions. The inner data can be mutated, but do not
    change the shape of the array after instantiating the class.

    Examples
    --------
    TODO

    See Also
    --------
    TODO
    '''

    def __new__(
        cls,
        pixels_array,
        xlim,
        ylim,
    ):
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

        Raises
        ------
        ValueError
            If `pixels_array` does not have exactly 3 dimensions or if
            `xlim` or `ylim` do not have exactly 2 values each.

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

        if xlim.ndim != 1 or len(xlim) != 2:
            raise ValueError(textwrap.fill((
                "The input `xlim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the pixel space in the x-dimension. "
                f"Received parameter with shape {xlim.shape}."
            )))

        ylim = np.asarray(ylim, dtype = np.float64)

        if ylim.ndim != 1 or len(ylim) != 2:
            raise ValueError(textwrap.fill((
                "The input `ylim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the pixel space in the y-dimension. "
                f"Received parameter with shape {ylim.shape}."
            )))

        # Setting class attributes
        pixels = pixels_array.view(cls)
        pixels._number_of_pixels = pixels.shape

        pixels._xlim = xlim
        pixels._ylim = ylim

        return pixels


    def __array_finalize__(self, pixels):
        # Required method for numpy subclassing
        if pixels is None:
            return

        self._xlim = getattr(pixels, "_xlim", None)
        self._ylim = getattr(pixels, "_ylim", None)


    def __reduce__(self):
        # __reduce__ and __setstate__ ensure correct pickling behaviour. See
        # https://stackoverflow.com/questions/26598109/preserve-custom-
        # attributes-when-pickling-subclass-of-numpy-array

        # Get the parent's __reduce__ tuple
        pickled_state = super(Pixels, self).__reduce__()

        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (
            self._xlim,
            self._ylim,
        )

        # Return a tuple that replaces the parent's __setstate__ tuple with
        # our own
        return (pickled_state[0], pickled_state[1], new_state)


    def __setstate__(self, state):
        # __reduce__ and __setstate__ ensure correct pickling behaviour
        # https://stackoverflow.com/questions/26598109/preserve-custom-
        # attributes-when-pickling-subclass-of-numpy-array

        # Set the class attributes
        self._ylim = state[-1]
        self._xlim = state[-2]

        # Call the parent's __setstate__ with the other tuple elements.
        super(Pixels, self).__setstate__(state[0:-2])


    @property
    def pixels(self):
        return self.__array__()


    @property
    def xlim(self):
        return self._xlim


    @property
    def ylim(self):
        return self._ylim


    @property
    def pixel_size(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_pixel_size"):
            self._pixel_size = np.array([
                (self.xlim[1] - self.xlim[0]) / self.shape[0],
                (self.ylim[1] - self.ylim[0]) / self.shape[1],
            ])

        return self._pixel_size


    @property
    def pixel_grid(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_pixel_grid"):
            self._pixel_grid = [
                np.linspace(lim[0], lim[1], self.shape[i] + 1)
                for i, lim in enumerate((self.xlim, self.ylim))
            ]

        return self._pixel_grid


    @property
    def pixel_lower(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_pixel_lower"):
            self._pixel_lower = np.array([self.xlim[0], self.ylim[0]])

        return self._pixel_lower


    @property
    def pixel_upper(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_pixel_upper"):
            self._pixel_upper = np.array([self.xlim[1], self.ylim[1]])

        return self._pixel_upper


    @staticmethod
    def zeros(shape, xlim, ylim):
        shape = tuple(shape)
        if len(shape) != 2:
            raise ValueError("The input `shape` must have two dimensions.")

        xlim = np.asarray(xlim, dtype = float)
        if xlim.ndim != 1 or xlim.shape[0] != 2:
            raise ValueError("`xlim` must have two floating-point values.")

        ylim = np.asarray(ylim, dtype = float)
        if ylim.ndim != 1 or ylim.shape[0] != 2:
            raise ValueError("`ylim` must have two floating-point values.")

        return Pixels(np.zeros(shape, dtype = float), xlim, ylim)


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
        Pixellise an array of lines and add them to a `PlotlyGrapher2D`
        instance:

        >>> lines = np.array(...)                   # shape (N, M >= 7)
        >>> lines2d = lines[:, [0, 1, 2, 4, 5]]     # select x, y of lines
        >>> number_of_pixels = [10, 10]
        >>> pixels = pept.Pixels.from_lines(lines2d, number_of_pixels)

        >>> grapher = pept.visualisation.PlotlyGrapher2D()
        >>> grapher.add_pixels(pixels)
        >>> grapher.show()

        Or add them directly to a raw `plotly.graph_objs` figure:

        >>> import plotly.graph_objs as go
        >>> fig = go.Figure()
        >>> fig.add_trace(pixels.heatmap_trace())
        >>> fig.show()

        '''

        # Compute the pixel centres
        x = self.pixel_grid[0]
        x = (x[1:] + x[:-1]) / 2

        y = self.pixel_grid[1]
        y = (y[1:] + y[:-1]) / 2

        heatmap = dict(
            x = x,
            y = y,
            z = self,
            colorscale = colorscale,
            transpose = transpose,
            xgap = xgap,
            ygap = ygap,
        )

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

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
        else:
            fig = plt.gcf()

        # Plot the values in pixels (this class is a numpy array subclass)
        ax.imshow(np.rot90(self))

        # Compute the pixel centres and set them in the Matplotlib image
        x = self.pixel_grid[0]
        x = (x[1:] + x[:-1]) / 2

        y = self.pixel_grid[1]
        y = (y[1:] + y[:-1]) / 2

        # Matplotlib shows numbers in a long format ("102.000032411"), so round
        # them to two decimals before plotting
        ax.set_xticklabels(np.round(x, 2))
        ax.set_yticklabels(np.round(y, 2))

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")

        return fig, ax


    def __str__(self):
        # Shown when calling print(class)
        docstr = (
            f"{self.__array__()}\n\n"
            f"shape =               {self.shape}\n"
            f"pixel_size =          {self.pixel_size}\n\n"
            f"xlim =                {self.xlim}\n"
            f"ylim =                {self.ylim}\n"
            f"pixel_grid:\n"
            f"([{self.pixel_grid[0][0]} ... {self.pixel_grid[0][-1]}],\n"
            f" [{self.pixel_grid[1][0]} ... {self.pixel_grid[1][-1]}])"
        )

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPR
        docstr = (
            "Class instance that inherits from `konigcell.Pixels`.\n"
            f"Type:\n{type(self)}\n\n"
            "Attributes\n----------\n"
            f"{self.__str__()}"
        )

        return docstr
