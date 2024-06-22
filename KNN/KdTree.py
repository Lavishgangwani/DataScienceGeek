import numpy as np

import pickle
from sklearn.neighbors import KDTree

rng = np.random.RandomState(0)

X = rng.random_sample((10, 3))  # 10 points in 3 dimensions

tree = KDTree(X, leaf_size=2)        

s = pickle.dumps(tree)                     

tree_copy = pickle.loads(s)                

dist, ind = tree_copy.query(X[:1], k=3)     

print(ind)  # indices of 3 closest neighbors
print(dist)  # distances to 3 closest neighbors