import pandas as pd
from pymds.mds import DistanceMatrix

# Distances between the vertices of a right-angled triangle
dist = pd.DataFrame({
    'a': [0.0, 1.0, 2.0],
    'b': [1.0, 0.0, 3 ** 0.5],
    'c': [2.0, 3 ** 0.5, 0.0]},
    index=['a', 'b', 'c'])

# Make an instance of DistanceMatrix
dm = DistanceMatrix(dist)

# Embed vertices in two dimensions
projection = dm.optimise(n=2)
