#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle as pickle
import gudhi
from pylab import *

# Load point cloud of nanotube
point_cloud = np.loadtxt("data/xyz/nt-6-6-10.xyz", skiprows=2, usecols=(1,2,3))
print(f"Loaded {len(point_cloud)} atoms")

# segment selection
z_sorted = point_cloud[np.argsort(point_cloud[:, 2])]
#segment = z_sorted[:72]
mid_start = (240 - 72) // 2
mid_end   = mid_start + 72
segment = z_sorted[mid_start:mid_end]
print(f"Selected segment from atom {mid_start} to {mid_end}, total {len(segment)} atoms")

# rips complex and persistence
rips_complex = gudhi.RipsComplex(points=segment, max_edge_length=8)
simplex_tree = rips_complex.create_simplex_tree(max_dimension=3)
#print(simplex_tree.num_simplices())

diag = simplex_tree.persistence(min_persistence=0.02)
# calculate bar counts per dimension
dims = [d for d, _ in diag]
for d in sorted(set(dims)):
    print(f"Dimension {d}: {dims.count(d)} bars")

# plot barcode
gudhi.plot_persistence_barcode(diag)
#plt.savefig("nano.png", dpi=300)
plt.show()


# result:
# Loaded 240 atoms
# Selected segment from atom 84 to 156, total 72 atoms
# Dimension 0: 72 bars
# Dimension 1: 25 bars
# Dimension 2: 47 bars