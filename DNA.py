import numpy as np

from utils.load_cif import load_cif
from utils.barcode_utils import barcode_from_point_cloud

# 2M45
def load_dna(mode: str) -> np.ndarray:
    coords = load_cif("data/biomole/2M54.cif", "dna", mode=mode)
    return coords

coords_all = load_dna("all")
coords_heavy = load_dna("heavy")
coords_P = load_dna("P")

for coords in [coords_all, coords_heavy, coords_P]:
    barcode_from_point_cloud(coords, 10, 2, separate_h0_h1=False, save_fig=f"figs/biomole/dna_{coords}_barcode.png", dpi=500)

