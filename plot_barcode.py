#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pickle
import gudhi
from pylab import *


point_cloud = np.loadtxt("c20.xyz", skiprows=1, usecols=(2,3,4))

rips_complex = gudhi.RipsComplex(points=point_cloud, max_edge_length=5)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
#print(simplex_tree.num_simplices())
diag = simplex_tree.persistence(min_persistence=0.02)

gudhi.plot_persistence_barcode(diag)
plt.savefig("barcode.png", dpi=300)
plt.show()