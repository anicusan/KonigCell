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


class Voxels:
    '''A class managing a 3D voxel space with physical dimensions, including
    tools for voxel manipulation and visualisation.

    The `.voxels` attribute is simply a `numpy.ndarray[ndim=3, dtype=float64]`.
    The `.attrs` dictionary can be used to store extra information.

    Attributes
    ----------
    voxels: (M, N, P) np.ndarray[ndim=3, dtype=float64]
        The 3D numpy array containing the voxel values. This class assumes a
        uniform grid of voxels - that is, the voxel size in each dimension is
        constant, but can vary from one dimension to another.

    xlim: (2,) np.ndarray[ndim=1, dtype=float64]
        The lower and upper boundaries of the voxellised volume in the
        x-dimension, formatted as [x_min, x_max].

    ylim: (2,) np.ndarray[ndim=1, dtype=float64]
        The lower and upper boundaries of the voxellised volume in the
        y-dimension, formatted as [y_min, y_max].

    zlim: (2,) np.ndarray[ndim=1, dtype=float64]
        The lower and upper boundaries of the voxellised volume in the
        z-dimension, formatted as [z_min, z_max].

    voxel_size: (3,) np.ndarray[ndim=1, dtype=float64]
        The lengths of a voxel in the x-, y- and z-dimensions, respectively.

    voxel_grids: (3,) list[np.ndarray[ndim=1, dtype=float64]]
        A list containing the voxel gridlines in the x-, y-, and z-dimensions.
        Each dimension's gridlines are stored as a numpy of the voxel
        delimitations, such that it has length (M + 1), where M is the number
        of voxels in given dimension.

    lower: (3,) np.ndarray[ndim=1, dtype=float64]
        The lower left corner of the voxel box; corresponds to
        [xlim[0], ylim[0], zlim[0]].

    upper: (3,) np.ndarray[ndim=1, dtype=float64]
        The upper right corner of the voxel box; corresponds to
        [xlim[1], ylim[1], zlim[1]].

    attrs: dict[Any, Any]
        A dictionary storing any other user-defined information.

    See Also
    --------
    konigcell.Pixels : a class managing a physical 2D pixel space.
    konigcell.dynamic3d : rasterize moving particles' trajectories.
    konigcell.static3d : rasterize static particles' positions.
    konigcell.dynamic_prob3d : 3D probability distribution of a quantity.
    '''
    __slots__ = ("_voxels", "_xlim", "_ylim", "_zlim", "_attrs", "_voxel_size",
                 "_voxel_grids", "_lower", "_upper")

    def __init__(self, voxels_array, xlim, ylim, zlim, **kwargs):
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

        kwargs: extra keyword arguments
            Extra user-defined attributes to be saved in `.attrs`.

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
        self._voxels = voxels_array
        self._xlim = xlim
        self._ylim = ylim
        self._zlim = zlim
        self._attrs = dict(kwargs)


    @property
    def voxels(self):
        return self._voxels


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
                (self._xlim[1] - self._xlim[0]) / self._voxels.shape[0],
                (self._ylim[1] - self._ylim[0]) / self._voxels.shape[1],
                (self._zlim[1] - self._zlim[0]) / self._voxels.shape[2],
            ])

        return self._voxel_size


    @property
    def voxel_grids(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_voxel_grids"):
            self._voxel_grids = [
                np.linspace(lim[0], lim[1], self._voxels.shape[i] + 1)
                for i, lim in enumerate((self._xlim, self._ylim, self._zlim))
            ]

        return self._voxel_grids


    @property
    def lower(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_lower"):
            self._lower = np.array([
                self._xlim[0],
                self._ylim[0],
                self._zlim[0],
            ])

        return self._lower


    @property
    def upper(self):
        # Compute once upon the first access and cache
        if not hasattr(self, "_upper"):
            self._upper = np.array([
                self._xlim[1],
                self._ylim[1],
                self._zlim[1],
            ])

        return self._upper


    @property
    def attrs(self):
        return self._attrs


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


    def copy(self, voxels_array = None, xlim = None, ylim = None, zlim = None,
             **kwargs):
        '''Create a copy of the current `Voxels` instance, optionally with new
        `voxels_array`, `xlim` and / or `ylim`.

        The extra attributes in `.attrs` are propagated too. Pass new
        attributes as extra keyword arguments.
        '''
        if voxels_array is None:
            voxels_array = self.voxels.copy()
        if xlim is None:
            xlim = self.xlim.copy()
        if ylim is None:
            ylim = self.ylim.copy()
        if zlim is None:
            zlim = self.zlim.copy()

        # Propagate attributes
        kwargs.update(self.attrs)

        return Voxels(voxels_array, xlim, ylim, zlim, **kwargs)


    @staticmethod
    def zeros(shape, xlim, ylim, zlim, **kwargs):
        zero_voxels = np.zeros(shape, dtype = float)
        return Voxels(zero_voxels, xlim, ylim, zlim, **kwargs)


    def from_physical(self, locations, corner = False):
        '''Transform `locations` from physical dimensions to voxel indices. If
        `corner = True`, return the index of the bottom left corner of each
        voxel; otherwise, use the voxel centres.

        Examples
        --------
        Create a simple `konigcell.Voxels` grid, spanning [-5, 5] mm in the
        X-dimension, [10, 20] mm in the Y-dimension and [0, 10] in Z:

        >>> import konigcell as kc
        >>> voxels = kc.Voxels.zeros((5, 5, 5), xlim=[-5, 5], ylim=[10, 20],
                                     zlim=[0, 10])
        >>> voxels
        Voxels
        ------
        xlim = [-5.  5.]
        ylim = [10. 20.]
        zlim = [10. 20.]
        voxels =
          (shape: (5, 5, 5))
          [[[0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]
            ...
            [0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]]
           [[0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]
            ...
            [0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]]
           ...
           [[0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]
            ...
            [0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]]
           [[0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]
            ...
            [0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]]]
        attrs = {}

        >>> voxels.voxel_size
        array([2., 2., 2.])

        Transform physical coordinates to voxel coordinates:

        >>> voxels.from_physical([-5, 10, 0], corner = True)
        array([0., 0., 0.])

        >>> voxels.from_physical([-5, 10, 0])
        array([-0.5, -0.5, -0.5])

        The voxel coordinates are returned exactly, as real numbers. For voxel
        indices, round them into values:

        >>> voxels.from_physical([0, 15, 0]).astype(int)
        array([2, 2, 0])

        Multiple coordinates can be given as a 2D array / list of lists:

        >>> voxels.from_physical([[0, 15, 0], [5, 20, 10]])
        array([[ 2. ,  2. , -0.5],
               [ 4.5,  4.5,  4.5]])

        '''

        offset = 0. if corner else self.voxel_size / 2
        return (locations - self.lower - offset) / self.voxel_size


    def to_physical(self, indices, corner = False):
        '''Transform `indices` from voxel indices to physical dimensions. If
        `corner = True`, return the coordinates of the bottom left corner of
        each voxel; otherwise, use the voxel centres.

        Examples
        --------
        Create a simple `konigcell.Voxels` grid, spanning [-5, 5] mm in the
        X-dimension, [10, 20] mm in the Y-dimension and [0, 10] in Z:

        >>> import konigcell as kc
        >>> voxels = kc.Voxels.zeros((5, 5, 5), xlim=[-5, 5], ylim=[10, 20],
                                     zlim=[0, 10])
        >>> voxels
        Voxels
        ------
        xlim = [-5.  5.]
        ylim = [10. 20.]
        zlim = [10. 20.]
        voxels =
          (shape: (5, 5, 5))
          [[[0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]
            ...
            [0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]]
           [[0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]
            ...
            [0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]]
           ...
           [[0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]
            ...
            [0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]]
           [[0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]
            ...
            [0. 0. ... 0. 0.]
            [0. 0. ... 0. 0.]]]
        attrs = {}

        >>> voxels.voxel_size
        array([2., 2., 2.])

        Transform physical coordinates to voxel coordinates:

        >>> voxels.to_physical([0, 0, 0], corner = True)
        array([-5., 10., 0.])

        >>> voxels.to_physical([0, 0, 0])
        array([-4., 11., 1.])

        Multiple coordinates can be given as a 2D array / list of lists:

        >>> voxels.to_physical([[0, 0, 0], [4, 4, 3]])
        array([[-4., 11.,  1.],
               [ 4., 19.,  7.]])

        '''

        offset = 0. if corner else self.voxel_size / 2
        return self.lower + indices * self.voxel_size + offset


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

        filtered_indices = np.argwhere(condition(self.voxels))
        positions = self.voxel_size * (0.5 + filtered_indices) + \
            [self.xlim[0], self.ylim[0], self.zlim[0]]

        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        voxel_vals = np.array([
            self.voxels[tuple(fi)]
            for fi in filtered_indices
        ])

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
        mode = "box",
        colorscale = "magma",
    ):
        # Type-checking inputs
        mode = str(mode).lower()

        vox = self.voxels.copy(order = "F")
        vox[~(condition(vox))] = 0.

        # You need to install PyVista to use this function!
        grid = pv.UniformGrid()
        grid.dimensions = np.array(vox.shape) + 1
        grid.origin = self.lower
        grid.spacing = self.voxel_size
        grid.cell_data["values"] = vox.flatten(order="F")

        # Create PyVista volumetric / voxel plot with an interactive clipper
        fig = pv.Plotter()

        if mode == "plane":
            fig.add_mesh_clip_plane(grid, cmap = colorscale)
        elif mode == "box":
            fig.add_mesh_clip_box(grid, cmap = colorscale)
        elif mode == "slice":
            fig.add_mesh(grid.slice_orthogonal(), cmap = colorscale)
        else:
            raise ValueError(textwrap.fill((
                "The input `mode` must be one of 'plane' | 'box' | 'slice'. "
                f"Received `mode={mode}`."
            )))

        return fig


    def vtk(
        self,
        condition = lambda voxels: voxels != 0.,
    ):
        vox = self.voxels.copy(order = "F")
        vox[~(condition(vox))] = 0.

        # You need to install PyVista to use this function!
        grid = pv.UniformGrid()
        grid.dimensions = np.array(vox.shape) + 1
        grid.origin = self.lower
        grid.spacing = self.voxel_size
        grid.cell_data["values"] = vox.flatten(order="F")

        return grid


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
            c = cmap(self.voxels[tuple(index)] / (self.voxels.max() or 1))
            cube.update(color = f"rgb({c[0]},{c[1]},{c[2]})")

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

        indices = np.argwhere(condition(self.voxels))
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

        filtered_indices = np.argwhere(condition(self.voxels))
        positions = self.voxel_size * (0.5 + filtered_indices) + \
            [self.xlim[0], self.ylim[0], self.zlim[0]]

        marker = dict(
            size = size,
            color = color,
            symbol = "square",
        )

        if colorbar and color is None:
            voxel_vals = [self.voxels[tuple(fi)] for fi in filtered_indices]
            marker.update(colorscale = colorscale, color = voxel_vals)

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
            z = self.voxels[ix, :, :]

            for i in range(1, width + 1):
                z = z + self.voxels[ix + i, :, :]
                z = z + self.voxels[ix - i, :, :]

        elif iy is not None:
            x = self.voxel_grids[0]
            y = self.voxel_grids[2]
            z = self.voxels[:, iy, :]

            for i in range(1, width + 1):
                z = z + self.voxels[:, iy + i, :]
                z = z + self.voxels[:, iy - i, :]

        elif iz is not None:
            x = self.voxel_grids[0]
            y = self.voxel_grids[1]
            z = self.voxels[:, :, iz]

            for i in range(1, width + 1):
                z = z + self.voxels[:, :, iz + i]
                z = z + self.voxels[:, :, iz - i]

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


    def __getitem__(self, *args, **kwargs):
        return self.pixels.__getitem__(*args, **kwargs)


    def __repr__(self):
        # String representation of the class
        name = "Voxels"
        underline = "-" * len(name)

        # Custom printing of the .lines and .samples_indices arrays
        with np.printoptions(threshold = 5, edgeitems = 2):
            voxels_str = f"{textwrap.indent(str(self.voxels), '  ')}"

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
            f"zlim = {self.ylim}\n"
            f"voxels = \n"
            f"  (shape: {self.voxels.shape})\n"
            f"{voxels_str}\n"
            "attrs = {"
            f"{attrs_str}"
            "}\n"
        )
