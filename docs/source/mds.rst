MDS
***

**Parameters and variables**

`m`: Number of samples. This is the number of rows and columns in `D` which must match.

`n`: Number of dimensions the multidimensional scaling result is embedded in.

`D`: The distance matrix. Shape (`m`, `m`)

`coords`: Coordinates of the samples embeded in `m`-dimensional space. Shape (`m`, `n`).

`x`: Coordinates of the samples embeded in `m`-dimensional space. Shape (`m` * `n`, ).

`d`: Pairwise distances between samples in `coords`. Shape (`m`, `m`).

`g`: Gradient of the error function at each coordinate in `coords`. Shape (`m`, `n`).

`diff`: `D` `-` `d`.

Distance Matrix
===============

.. autoclass:: pymds.DistanceMatrix
   :members:

Projection
==========

.. autoclass:: pymds.Projection
   :members:
