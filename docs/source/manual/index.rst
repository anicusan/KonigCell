******
Manual
******

All public ``konigcell`` subroutines are fully documented here, along with copy-pastable
examples. The `base` functionality is summarised below. You can also use the
`Search` bar in the top left to go directly to what you need.

We really appreciate all help with writing useful documentation; if you feel
something can be improved, or would like to share some example code, by all means
get in contact with us - or be a superhero and click `Edit this page` on the right
and submit your changes to the GitHub repository directly!


Grids
-----

.. autosummary::
   :toctree: generated/

   konigcell.Pixels
   konigcell.Voxels


Projection Modes
----------------
``konigcell.RATIO`` - Weight cell values by the ratio of the intersected cell area /
volume and the total circle area / sphere volume.

``konigcell.INTERSECTION`` - Weight cell values by their intersected area / volume with
the rasterized shape.

``konigcell.PARTICLE`` - Weight cell values by the rasterized shape's area / volume.

``konigcell.ONE`` - Add factors to cells with no weighting.


2D Projections
--------------

.. autosummary::
   :toctree: generated/

   konigcell.dynamic2d
   konigcell.static2d
   konigcell.dynamic_prob2d
   konigcell.static_prob2d


3D Projections
--------------

.. autosummary::
   :toctree: generated/

   konigcell.dynamic3d
   konigcell.static3d
   konigcell.dynamic_prob3d
   konigcell.static_prob3d


Low Level Interface
-------------------

.. autosummary::
   :toctree: generated/

   konigcell.kc2d
   konigcell.kc3d


Plotting Utilities
------------------

.. autosummary::
   :toctree: generated/

   konigcell.create_fig
   konigcell.format_fig

