import numpy as np

from utils.load_cif import load_cif, count_models, count_atoms_in_model
from utils.barcode_utils import barcode_from_point_cloud

# 2M45
def load_dna(mode: str) -> np.ndarray:
    coords = load_cif("data/biomole/2M54.cif", "dna", mode=mode, model_id=0)
    return coords

count_models("data/biomole/2M54.cif", "dna")

count_atoms_in_model("data/biomole/2M54.cif", "dna", model_id=2)

dna_all = load_dna("all")
dna_heavy = load_dna("heavy")
dna_P = load_dna("P")

dna_name_dict = {
    "all": dna_all,
    "heavy": dna_heavy,
    "P": dna_P
}

for dna_name, coords in dna_name_dict.items():
    barcode_from_point_cloud(coords, 10, 2, separate_h0_h1=False, save_fig=f"figs/biomole/dna_{dna_name}_barcode.png", dpi=500)