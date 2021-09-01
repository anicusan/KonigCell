#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : voxels.py
# License: MIT
# Author : Andrei Leonard Nicusan <a.l.nicusan@bham.ac.uk>
# Date   : 20.05.2021


import  pickle
import  textwrap

import  numpy                       as  np

# Plotting is optional
try:
    import  plotly.graph_objects    as  go
except ImportError:
    pass

try:
    import  matplotlib
    import  matplotlib.pyplot       as  plt
except ImportError:
    pass

try:
    import  pyvista                 as  pv
except ImportError:
    pass


class Voxels(np.ndarray):
    '''A `numpy.ndarray` subclass that manages a 3D voxel space, including
    tools for voxel manipulation and visualisation.

    Besides a 3D array of voxels, this class also stores the physical space
    spanned by the voxel grid `xlim`, `ylim` and `zlim` along with some
    voxel-specific helper methods.

    This subclasses the `numpy.ndarray` class, so any `Voxels` object acts
    exactly like a 3D numpy array. All numpy methods and operations are valid
    on `Voxels` (e.g. add 1 to all voxels with `voxels += 1`).

    Attributes
    ----------
    voxels: (M, N, P) numpy.ndarray
        The 3D numpy array containing the number of lines that pass through
        each voxel. They are stored as `float`s. This class assumes a uniform
        grid of voxels - that is, the voxel size in each dimension is constant,
        but can vary from one dimension to another. The number of voxels in
        each dimension is defined by `number_of_voxels`.

    xlim: (2,) numpy.ndarray
        The lower and upper boundaries of the voxellised volume in the
        x-dimension, formatted as [x_min, x_max].

    ylim: (2,) numpy.ndarray
        The lower and upper boundaries of the voxellised volume in the
        y-dimension, formatted as [y_min, y_max].

    zlim: (2,) numpy.ndarray
        The lower and upper boundaries of the voxellised volume in the
        z-dimension, formatted as [z_min, z_max].

    voxel_size: (3,) numpy.ndarray
        The lengths of a voxel in the x-, y- and z-dimensions, respectively.

    voxel_grids: list[numpy.ndarray]
        A list containing the voxel gridlines in the x-, y-, and z-dimensions.
        Each dimension's gridlines are stored as a numpy of the voxel
        delimitations, such that it has length (M + 1), where M is the number
        of voxels in given dimension.

    voxel_lower: numpy.ndarray
        The lower left corner of the voxel box.

    voxel_upper: numpy.ndarray
        The upper right corner of the voxel box.

    Examples
    --------
    This class is most often instantiated from a sample of lines to voxellise:

    >>> import pept
    >>> import numpy as np

    >>> lines = np.arange(70).reshape(10, 7)

    >>> number_of_voxels = [3, 4, 5]
    >>> voxels = pept.Voxels.from_lines(lines, number_of_voxels)
    >>> Initialised Voxels class in 0.0006861686706542969 s.

    >>> print(voxels)
    >>> voxels:
    >>> [[[2. 1. 0. 0. 0.]
    >>>   [0. 2. 0. 0. 0.]
    >>>   [0. 0. 0. 0. 0.]
    >>>   [0. 0. 0. 0. 0.]]

    >>>  [[0. 0. 0. 0. 0.]
    >>>   [0. 1. 1. 0. 0.]
    >>>   [0. 0. 1. 1. 0.]
    >>>   [0. 0. 0. 0. 0.]]

    >>>  [[0. 0. 0. 0. 0.]
    >>>   [0. 0. 0. 0. 0.]
    >>>   [0. 0. 0. 2. 0.]
    >>>   [0. 0. 0. 1. 2.]]]

    >>> number_of_voxels =    (3, 4, 5)
    >>> voxel_size =          [22.  16.5 13.2]

    >>> xlim =                [ 1. 67.]
    >>> ylim =                [ 2. 68.]
    >>> zlim =                [ 3. 69.]

    >>> voxel_grids:
    >>> [array([ 1., 23., 45., 67.]),
    >>>  array([ 2. , 18.5, 35. , 51.5, 68. ]),
    >>>  array([ 3. , 16.2, 29.4, 42.6, 55.8, 69. ])]

    Note that it is important to define the `number_of_voxels`.

    See Also
    --------
    pept.VoxelData : Asynchronously manage multiple voxel spaces.
    pept.LineData : Encapsulate lines for ease of iteration and plotting.
    pept.PointData : Encapsulate points for ease of iteration and plotting.
    PlotlyGrapher : Easy, publication-ready plotting of PEPT-oriented data.
    '''

    def __new__(
        cls,
        voxels_array,
        xlim,
        ylim,
        zlim,
    ):
        '''`Voxels` class constructor.

        Parameters
        ----------
        voxels_array: 3D numpy.ndarray
            A 3D numpy array, corresponding to a pre-defined voxel space.

        xlim: (2,) numpy.ndarray
            The lower and upper boundaries of the voxellised volume in the
            x-dimension, formatted as [x_min, x_max].

        ylim: (2,) numpy.ndarray
            The lower and upper boundaries of the voxellised volume in the
            y-dimension, formatted as [y_min, y_max].

        zlim: (2,) numpy.ndarray
            The lower and upper boundaries of the voxellised volume in the
            z-dimension, formatted as [z_min, z_max].

        Raises
        ------
        ValueError
            If `voxels_array` does not have exactly 3 dimensions or if
            `xlim`, `ylim` or `zlim` do not have exactly 2 values each.

        '''

        # Type-checking inputs
        voxels_array = np.asarray(
            voxels_array,
            order = "C",
            dtype = float
        )

        if voxels_array.ndim != 3:
            raise ValueError(textwrap.fill((
                "The input `voxels_array` must contain an array-like with "
                "exactly three dimensions (i.e. pre-made voxels array). "
                f"Received an array with {voxels_array.ndim} dimensions."
            )))

        xlim = np.asarray(xlim, dtype = float)

        if xlim.ndim != 1 or len(xlim) != 2:
            raise ValueError(textwrap.fill((
                "The input `xlim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the voxel space in the x-dimension. "
                f"Received parameter with shape {xlim.shape}."
            )))

        ylim = np.asarray(ylim, dtype = float)

        if ylim.ndim != 1 or len(ylim) != 2:
            raise ValueError(textwrap.fill((
                "The input `ylim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the voxel space in the y-dimension. "
                f"Received parameter with shape {ylim.shape}."
            )))

        zlim = np.asarray(zlim, dtype = float)

        if zlim.ndim != 1 or len(zlim) != 2:
            raise ValueError(textwrap.fill((
                "The input `zlim` parameter must be a list with exactly "
                "two values, corresponding to the minimum and maximum "
                "coordinates of the voxel space in the z-dimension. "
                f"Received parameter with shape {zlim.shape}."
            )))

        # Setting class attributes
        voxels = voxels_array.view(cls)

        voxels._xlim = xlim
        voxels._ylim = ylim
        voxels._zlim = zlim

        '''
        voxels._voxel_size = np.array([
            (voxels._xlim[1] - voxels._xlim[0]) / voxels._number_of_voxels[0],
            (voxels._ylim[1] - voxels._ylim[0]) / voxels._number_of_voxels[1],
            (voxels._zlim[1] - voxels._zlim[0]) / voxels._number_of_voxels[2],
        ])

        voxels._voxel_grids = tuple([
            np.linspace(lim[0], lim[1], voxels._number_of_voxels[i] + 1)
            for i, lim in enumerate((voxels._xlim, voxels._ylim, voxels._zlim))
        ])
        '''

        return voxels


    def __array_finalize__(self, voxels):
        if voxels is None:
            return

        self._xlim = getattr(voxels, "_xlim", None)
        self._ylim = getattr(voxels, "_ylim", None)
        self._zlim = getattr(voxels, "_zlim", None)


    def __reduce__(self):
        # __reduce__ and __setstate__ ensure correct pickling behaviour. See
        # https://stackoverflow.com/questions/26598109/preserve-custom-
        # attributes-when-pickling-subclass-of-numpy-array

        # Get the parent's __reduce__ tuple
        pickled_state = super(Voxels, self).__reduce__()

        # Create our own tuple to pass to __setstate__
        new_state = pickled_state[2] + (
            self._xlim,
            self._ylim,
            self._zlim,
        )

        # Return a tuple that replaces the parent's __setstate__ tuple with
        # our own
        return (pickled_state[0], pickled_state[1], new_state)


    def __setstate__(self, state):
        # __reduce__ and __setstate__ ensure correct pickling behaviour
        # https://stackoverflow.com/questions/26598109/preserve-custom-
        # attributes-when-pickling-subclass-of-numpy-array

        # Set the class attributes
        self._zlim = state[-1]
        self._ylim = state[-2]
        self._xlim = state[-3]

        # Call the parent's __setstate__ with the other tuple elements.
        super(Voxels, self).__setstate__(state[0:-3])


    @property
    def voxels(self):
        return self.__array__()


    @property
    def xlim(self):
        return self._xlim


    @property
    def ylim(self):
        return self._ylim


    @property
    def zlim(self):
        return self._zlim


    @property
    def voxel_size(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_voxel_size"):
            self._voxel_size = np.array([
                (self._xlim[1] - self._xlim[0]) / self.shape[0],
                (self._ylim[1] - self._ylim[0]) / self.shape[1],
                (self._zlim[1] - self._zlim[0]) / self.shape[2],
            ])

        return self._voxel_size


    @property
    def voxel_grid(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_voxel_grid"):
            self._voxel_grid = [
                np.linspace(lim[0], lim[1], self.shape[i] + 1)
                for i, lim in enumerate((self.xlim, self.ylim, self.zlim))
            ]

        return self._voxel_grid


    @property
    def voxel_lower(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_voxel_lower"):
            self._voxel_lower = np.array([
                self.xlim[0],
                self.ylim[0],
                self.zlim[0],
            ])

        return self._voxel_lower


    @property
    def voxel_upper(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_voxel_upper"):
            self._voxel_upper = np.array([
                self.xlim[1],
                self.ylim[1],
                self.zlim[1],
            ])

        return self._voxel_upper


    def save(self, filepath):
        '''Save a `Voxels` instance as a binary `pickle` object.

        Saves the full object state, including the inner `.voxels` NumPy array,
        `xlim`, etc. in a fast, portable binary format. Load back the object
        using the `load` method.

        Parameters
        ----------
        filepath : filename or file handle
            If filepath is a path (rather than file handle), it is relative
            to where python is called.

        Examples
        --------
        Save a `Voxels` instance, then load it back:

        >>> import numpy as np
        >>> import konigcell as kc
        >>>
        >>> grid = np.zeros((64, 48, 32))
        >>> voxels = kc.Voxels(grid, [0, 20], [0, 10])
        >>> voxels.save("voxels.pickle")

        >>> voxels_reloaded = kc.Voxels.load("voxels.pickle")

        '''
        with open(filepath, "wb") as f:
            pickle.dump(self, f)


    @staticmethod
    def load(filepath):
        '''Load a saved / pickled `Voxels` object from `filepath`.

        Most often the full object state was saved using the `.save` method.

        Parameters
        ----------
        filepath : filename or file handle
            If filepath is a path (rather than file handle), it is relative
            to where python is called.

        Returns
        -------
        pept.Voxels
            The loaded `pept.Voxels` instance.

        Examples
        --------
        Save a `Voxels` instance, then load it back:

        >>> import numpy as np
        >>> import konigcell as kc
        >>>
        >>> grid = np.zeros((64, 48, 32))
        >>> voxels = kc.Voxels(grid, [0, 20], [0, 10])
        >>> voxels.save("voxels.pickle")

        >>> voxels_reloaded = kc.Voxels.load("voxels.pickle")

        '''
        with open(filepath, "rb") as f:
            obj = pickle.load(f)

        return obj


    @staticmethod
    def zeros(shape, xlim, ylim, zlim):
        shape = tuple(shape)
        if len(shape) != 3:
            raise ValueError("The input `shape` must have three dimensions.")

        xlim = np.asarray(xlim, dtype = float)
        if xlim.ndim != 1 or xlim.shape[0] != 2:
            raise ValueError("`xlim` must have two floating-point values.")

        ylim = np.asarray(ylim, dtype = float)
        if ylim.ndim != 1 or ylim.shape[0] != 2:
            raise ValueError("`ylim` must have two floating-point values.")

        zlim = np.asarray(zlim, dtype = float)
        if zlim.ndim != 1 or zlim.shape[0] != 2:
            raise ValueError("`zlim` must have two floating-point values.")


        return Voxels(np.zeros(shape, dtype = float), xlim, ylim, zlim)


    def plot(
        self,
        condition = lambda voxel_data: voxel_data > 0,
        ax = None,
        alt_axes = False,
    ):
        '''Plot the voxels in this class using Matplotlib.

        This plots the centres of all voxels encapsulated in a `pept.Voxels`
        instance, colour-coding the voxel value.

        The `condition` parameter is a filtering function that should return
        a boolean mask (i.e. it is the result of a condition evaluation). For
        example `lambda x: x > 0` selects all voxels that have a value larger
        than 0.

        Parameters
        ----------
        condition : function, default `lambda voxel_data: voxel_data > 0`
            The filtering function applied to the voxel data before plotting
            it. It should return a boolean mask (a numpy array of the same
            shape, filled with True and False), selecting all voxels that
            should be plotted. The default, `lambda x: x > 0` selects all
            voxels which have a value larger than 0.

        ax : mpl_toolkits.mplot3D.Axes3D object, optional
            The 3D matplotlib-based axis for plotting. If undefined, new
            Matplotlib figure and axis objects are created.

        alt_axes : bool, default False
            If `True`, plot using the alternative PEPT-style axes convention:
            z is horizontal, y points upwards. Because Matplotlib cannot swap
            axes, this is achieved by swapping the parameters in the plotting
            call (i.e. `plt.plot(x, y, z)` -> `plt.plot(z, x, y)`).

        Returns
        -------
        fig, ax : matplotlib figure and axes objects

        Notes
        -----
        Plotting all points is very computationally-expensive for matplotlib.
        It is recommended to only plot a couple of samples at a time, or use
        Plotly, which is faster.

        Examples
        --------
        Voxellise an array of lines and add them to a `PlotlyGrapher` instance:

        >>> import konigcell as kc
        >>>
        >>> lines = np.array(...)           # shape (N, M >= 7)
        >>> number_of_voxels = [10, 10, 10]
        >>> voxels = kc.Voxels(lines, number_of_voxels)

        >>> fig, ax = voxels.plot()
        >>> fig.show()

        '''

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection = '3d')
        else:
            fig = plt.gcf()

        filtered_indices = np.argwhere(condition(self))
        positions = self.voxel_size * (0.5 + filtered_indices) + \
            [self.xlim[0], self.ylim[0], self.zlim[0]]

        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        voxel_vals = np.array([self[tuple(fi)] for fi in filtered_indices])

        cmap = plt.cm.magma
        color_array = cmap(voxel_vals / voxel_vals.max())

        if alt_axes:
            ax.scatter(z, x, y, c = color_array, marker = "s")

            ax.set_xlabel("z (mm)")
            ax.set_ylabel("x (mm)")
            ax.set_zlabel("y (mm)")

        else:
            ax.scatter(x, y, z, c = color_array, marker = "s")

            ax.set_xlabel("x (mm)")
            ax.set_ylabel("y (mm)")
            ax.set_zlabel("z (mm)")

        return fig, ax


    def plot_volumetric(
        self,
        condition = lambda voxels: voxels > 0,
    ):
        vox = self.voxels.copy(order = "F")
        vox[~(condition(vox))] = 0.

        # You need to install PyVista to use this function!
        grid = pv.UniformGrid()
        grid.dimensions = np.array(vox.shape) + 1
        grid.origin = self.voxel_lower
        grid.spacing = self.voxel_size
        grid.cell_arrays["values"] = vox.flatten(order="F")

        # Create PyVista volumetric / voxel plot with an interactive clipper
        fig = pv.Plotter()
        fig.add_mesh_clip_plane(grid)

        return fig


    def cube_trace(
        self,
        index,
        color = None,
        opacity = 0.4,
        colorbar = True,
        colorscale = "magma",
    ):
        '''Get the Plotly `Mesh3d` trace for a single voxel at `index`.

        This renders the voxel as a cube. While visually accurate, this method
        is *very* computationally intensive - only use it for fewer than 100
        cubes. For more voxels, use the `voxels_trace` method.

        Parameters
        ----------
        index: (3,) tuple
            The voxel indices, given as a 3-tuple.

        color : str or list-like, optional
            Can be a single color (e.g. "black", "rgb(122, 15, 241)") or a
            colorbar list. Overrides `colorbar` if set. For more information,
            check the Plotly documentation. The default is None.

        opacity : float, default 0.4
            The opacity of the lines, where 0 is transparent and 1 is fully
            opaque.

        colorbar : bool, default True
            If set to True, will color-code the voxel values. Is overridden if
            `color` is set.

        colorscale : str, default "Magma"
            The Plotly scheme for color-coding the voxel values in the input
            data. Typical ones include "Cividis", "Viridis" and "Magma".
            A full list is given at `plotly.com/python/builtin-colorscales/`.
            Only has an effect if `colorbar = True` and `color` is not set.

        Raises
        ------
        ValueError
            If `index` does not contain exactly three values.

        Notes
        -----
        If you want to render a small number of voxels as cubes using Plotly,
        use the `cubes_traces` method, which creates a list of individual cubes
        for all voxels, using this function.

        '''

        index = np.asarray(index, dtype = int)

        if index.ndim != 1 or len(index) != 3:
            raise ValueError(textwrap.fill((
                "The input `index` must contain exactly three values, "
                "corresponding to the x, y, z indices of the voxel to plot. "
                f"Received {index}."
            )))

        xyz = self.voxel_size * index + \
            [self.xlim[0], self.ylim[0], self.zlim[0]]

        x = np.array([0, 0, 1, 1, 0, 0, 1, 1]) * self.voxel_size[0]
        y = np.array([0, 1, 1, 0, 0, 1, 1, 0]) * self.voxel_size[1]
        z = np.array([0, 0, 0, 0, 1, 1, 1, 1]) * self.voxel_size[2]
        i = np.array([7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2])
        j = np.array([3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3])
        k = np.array([0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6])

        cube = dict(
            x =  x + xyz[0],
            y =  y + xyz[1],
            z =  z + xyz[2],
            i =  i,
            j =  j,
            k =  k,
            opacity = opacity,
            color = color
        )

        if colorbar and color is None:
            cmap = matplotlib.cm.get_cmap(colorscale)
            c = cmap(self[tuple(index)] / (self.max() or 1))
            cube.update(
                color = "rgb({},{},{})".format(c[0], c[1], c[2])
            )

        # You need to install Plotly to use this function!
        return go.Mesh3d(cube)


    def cubes_traces(
        self,
        condition = lambda voxels: voxels > 0,
        color = None,
        opacity = 0.4,
        colorbar = True,
        colorscale = "magma",
    ):
        '''Get a list of Plotly `Mesh3d` traces for all voxels selected by the
        `condition` filtering function.

        The `condition` parameter is a filtering function that should return
        a boolean mask (i.e. it is the result of a condition evaluation). For
        example `lambda x: x > 0` selects all voxels that have a value larger
        than 0.

        This renders each voxel as individual cubes. While visually accurate,
        this method is *very* computationally intensive - only use it for fewer
        than 100 cubes. For more voxels, use the `voxels_trace` method.

        Parameters
        ----------
        condition : function, default `lambda voxels: voxels > 0`
            The filtering function applied to the voxel data before plotting
            it. It should return a boolean mask (a numpy array of the same
            shape, filled with True and False), selecting all voxels that
            should be plotted. The default, `lambda x: x > 0` selects all
            voxels which have a value larger than 0.

        color : str or list-like, optional
            Can be a single color (e.g. "black", "rgb(122, 15, 241)") or a
            colorbar list. Overrides `colorbar` if set. For more information,
            check the Plotly documentation. The default is None.

        opacity : float, default 0.4
            The opacity of the lines, where 0 is transparent and 1 is fully
            opaque.

        colorbar : bool, default True
            If set to True, will color-code the voxel values. Is overridden if
            `color` is set.

        colorscale : str, default "magma"
            The Plotly scheme for color-coding the voxel values in the input
            data. Typical ones include "Cividis", "Viridis" and "Magma".
            A full list is given at `plotly.com/python/builtin-colorscales/`.
            Only has an effect if `colorbar = True` and `color` is not set.

        Examples
        --------
        Plot a `konigcell.Voxels` on a `plotly.graph_objs.Figure`.

        >>> import konigcell as kc
        >>> voxels = ...

        >>> import plotly.graph_objs as go
        >>>
        >>> fig = go.Figure()
        >>> fig.add_traces(voxels.cubes_traces())  # small number of voxels
        >>> fig.show()

        '''

        indices = np.argwhere(condition(self))
        traces = [
            self.cube_trace(
                i,
                color = color,
                opacity = opacity,
                colorbar = colorbar,
                colorscale = colorscale,
            ) for i in indices
        ]

        return traces


    def scatter_trace(
        self,
        condition = lambda voxel_data: voxel_data > 0,
        size = 4,
        color = None,
        opacity = 0.4,
        colorbar = True,
        colorscale = "Magma",
        colorbar_title = None,
    ):
        '''Create and return a trace for all the voxels in this class, with
        possible filtering.

        Creates a `plotly.graph_objects.Scatter3d` object for the centres of
        all voxels encapsulated in a `pept.Voxels` instance, colour-coding the
        voxel value.

        The `condition` parameter is a filtering function that should return
        a boolean mask (i.e. it is the result of a condition evaluation). For
        example `lambda x: x > 0` selects all voxels that have a value larger
        than 0.

        Parameters
        ----------
        condition : function, default `lambda voxel_data: voxel_data > 0`
            The filtering function applied to the voxel data before plotting
            it. It should return a boolean mask (a numpy array of the same
            shape, filled with True and False), selecting all voxels that
            should be plotted. The default, `lambda x: x > 0` selects all
            voxels which have a value larger than 0.

        size : float, default 4
            The size of the plotted voxel points. Note that due to the large
            number of voxels in typical applications, the *voxel centres* are
            plotted as square points, which provides an easy to understand
            image that is also fast and responsive.

        color : str or list-like, optional
            Can be a single color (e.g. "black", "rgb(122, 15, 241)") or a
            colorbar list. Overrides `colorbar` if set. For more information,
            check the Plotly documentation. The default is None.

        opacity : float, default 0.4
            The opacity of the lines, where 0 is transparent and 1 is fully
            opaque.

        colorbar : bool, default True
            If set to True, will color-code the voxel values. Is overridden if
            `color` is set.

        colorscale : str, default "Magma"
            The Plotly scheme for color-coding the voxel values in the input
            data. Typical ones include "Cividis", "Viridis" and "Magma".
            A full list is given at `plotly.com/python/builtin-colorscales/`.
            Only has an effect if `colorbar = True` and `color` is not set.

        colorbar_title : str, optional
            If set, the colorbar will have this title above it.

        Examples
        --------
        Voxellise an array of lines and add them to a `PlotlyGrapher` instance:

        >>> grapher = PlotlyGrapher()
        >>> lines = np.array(...)           # shape (N, M >= 7)
        >>> number_of_voxels = [10, 10, 10]
        >>> voxels = pept.Voxels.from_lines(lines, number_of_voxels)
        >>> grapher.add_lines(lines)
        >>> grapher.add_trace(voxels.voxels_trace())
        >>> grapher.show()

        '''

        filtered_indices = np.argwhere(condition(self))
        positions = self.voxel_size * (0.5 + filtered_indices) + \
            [self.xlim[0], self.ylim[0], self.zlim[0]]

        marker = dict(
            size = size,
            color = color,
            symbol = "square",
        )

        if colorbar and color is None:
            voxel_vals = [self[tuple(fi)] for fi in filtered_indices]
            marker.update(colorscale = "Magma", color = voxel_vals)

            if colorbar_title is not None:
                marker.update(colorbar = dict(title = colorbar_title))

        voxels = dict(
            x = positions[:, 0],
            y = positions[:, 1],
            z = positions[:, 2],
            opacity = opacity,
            mode = "markers",
            marker = marker,
        )

        # You need to install Plotly to use this function!
        return go.Scatter3d(voxels)


    def heatmap_trace(
        self,
        ix = None,
        iy = None,
        iz = None,
        width = 0,
        colorscale = "Magma",
        transpose = True
    ):
        '''Create and return a Plotly `Heatmap` trace of a 2D slice through the
        voxels.

        The orientation of the slice is defined by the input `ix` (for the YZ
        plane), `iy` (XZ), `iz` (XY) parameters - which correspond to the
        voxel index in the x-, y-, and z-dimension. Importantly, at least one
        of them must be defined.

        Parameters
        ----------
        ix : int, optional
            The index along the x-axis of the voxels at which a YZ slice is to
            be taken. One of `ix`, `iy` or `iz` must be defined.

        iy: int, optional
            The index along the y-axis of the voxels at which a XZ slice is to
            be taken. One of `ix`, `iy` or `iz` must be defined.

        iz : int, optional
            The index along the z-axis of the voxels at which a XY slice is to
            be taken. One of `ix`, `iy` or `iz` must be defined.

        width : int, default 0
            The number of voxel layers around the given slice index to collapse
            (i.e. accumulate) onto the heatmap.

        colorscale : str, default "Magma"
            The Plotly scheme for color-coding the voxel values in the input
            data. Typical ones include "Cividis", "Viridis" and "Magma".
            A full list is given at `plotly.com/python/builtin-colorscales/`.
            Only has an effect if `colorbar = True` and `color` is not set.

        transpose : bool, default True
            Transpose the heatmap (i.e. flip it across its diagonal).

        Raises
        ------
        ValueError
            If neither of `ix`, `iy` or `iz` was defined.

        Examples
        --------
        Voxellise an array of lines and add them to a `PlotlyGrapher` instance:

        >>> lines = np.array(...)           # shape (N, M >= 7)
        >>> number_of_voxels = [10, 10, 10]
        >>> voxels = pept.Voxels(lines, number_of_voxels)

        >>> import plotly.graph_objs as go
        >>> fig = go.Figure()
        >>> fig.add_trace(voxels.heatmap_trace())
        >>> fig.show()

        '''

        if ix is not None:
            x = self.voxel_grids[1]
            y = self.voxel_grids[2]
            z = self[ix, :, :]

            for i in range(1, width + 1):
                z = z + self[ix + i, :, :]
                z = z + self[ix - i, :, :]

        elif iy is not None:
            x = self.voxel_grids[0]
            y = self.voxel_grids[2]
            z = self[:, iy, :]

            for i in range(1, width + 1):
                z = z + self[:, iy + i, :]
                z = z + self[:, iy - i, :]

        elif iz is not None:
            x = self.voxel_grids[0]
            y = self.voxel_grids[1]
            z = self[:, :, iz]

            for i in range(1, width + 1):
                z = z + self[:, :, iz + i]
                z = z + self[:, :, iz - i]

        else:
            raise ValueError(textwrap.fill((
                "[ERROR]: One of the `ix`, `iy`, `iz` slice indices must be "
                "provided."
            )))

        heatmap = dict(
            x = x,
            y = y,
            z = z,
            colorscale = colorscale,
            transpose = transpose,
        )

        # You need to install Plotly to use this function!
        return go.Heatmap(heatmap)


    def __str__(self):
        # Shown when calling print(class)
        docstr = (
            f"{self.__array__()}\n\n"
            f"shape =               {self.shape}\n"
            f"voxel_size =          {self.voxel_size}\n\n"
            f"xlim =                {self.xlim}\n"
            f"ylim =                {self.ylim}\n"
            f"zlim =                {self.zlim}\n\n"
            f"voxel_grids:\n"
            f"([{self.voxel_grid[0][0]} ... {self.voxel_grid[0][-1]}],\n"
            f" [{self.voxel_grid[1][0]} ... {self.voxel_grid[1][-1]}],\n"
            f" [{self.voxel_grid[2][0]} ... {self.voxel_grid[2][-1]}])"
        )

        return docstr


    def __repr__(self):
        # Shown when writing the class on a REPR
        docstr = (
            "Class instance that inherits from `konigcell.Voxels`.\n"
            f"Type:\n{type(self)}\n\n"
            "Attributes\n----------\n"
            f"{self.__str__()}"
        )

        return docstr
