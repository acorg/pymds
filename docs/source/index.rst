#####
pymds
#####

Metric multidimensional scaling in python.

.. toctree::
   :maxdepth: 2

   mds

*******
Install
*******

Use `pip`_::

    pip install pymds

*******
Example
*******

Multidimensional scaling aims to embed samples as points in `n`-dimensional space, where the distances between points represent distances between samples in data.

In this example, edges of a triangle are specified by setting the distances between three vertices `a`, `b` and `c`. These data can be represented perfectly in 2-dimensional space. 

.. literalinclude:: distance-matrix-example.py

In datasets where the distances between samples cannot be represented perfectly in `n`-dimensional space residual error exists among the distances between samples in the space and the distances in the data.

Error in MDS is also known as `stress`.

.. _pip: https://pypi.python.org/pypi/pip
