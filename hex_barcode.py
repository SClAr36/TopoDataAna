#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pickle
import gudhi
from pylab import *

points = np.loadtxt("hex_points.txt", comments="#")

rips_complex = gudhi.RipsComplex(points=points, max_edge_length=10)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
#print(simplex_tree.num_simplices())
diag = simplex_tree.persistence(min_persistence=0.02)

gudhi.plot_persistence_barcode(diag)
plt.savefig("hex_barcode.png", dpi=500)
plt.show()