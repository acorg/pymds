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

In this example, three edges of a triangle are specified by setting the distances between vertices `a`, `b` and `c`. These data can be represented perfectly in 2-dimensional space. 

.. literalinclude:: distance-matrix-example.py

Some datasets cannot be represented perfectly like this meaning that residual error will exist among the distances between samples in the space and the distances in the data.

Error in MDS is also known as `stress`.

.. _pip: https://pypi.python.org/pypi/pip
