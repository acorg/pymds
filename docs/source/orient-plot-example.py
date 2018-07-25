from pymds import DistanceMatrix

from numpy.random import uniform, seed
from scipy.spatial.distance import pdist, squareform

import seaborn as sns
sns.set_style('whitegrid')

# 50 random 2D samples
seed(1234)
samples = uniform(low=-10, high=10, size=(50, 2))

# Measure pairwise distances between samples
dists = squareform(pdist(samples))

dists_shrunk = dists * 0.65

# Embed
original = DistanceMatrix(dists).embed()
shrunk = DistanceMatrix(dists_shrunk).embed()

shrunk.orient_to(original, inplace=True)

original.plot(c='black', edgecolor='white', s=50)
original.plot_lines_to(shrunk, linewidths=0.5, colors='black')
