import numpy as np

from utils.load_cif import load_cif
from utils.barcode_utils import barcode_from_point_cloud

def load_dna(mode: str) -> np.ndarray:
    coords = load_cif("data/biomole/2M54.cif", "dna", mode=mode)
    return coords

coords_all = load_dna("all")
coords_heavy = load_dna("heavy")
coords_P = load_dna("P")

barcode_from_point_cloud(coords_heavy, 10, 2)
