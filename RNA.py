import numpy as np

from utils.load_cif import load_cif, count_models, count_atoms_in_model
from utils.barcode_utils import barcode_from_point_cloud

count_models("data/biomole/4QG3.cif", "rna")
count_atoms_in_model("data/biomole/4QG3.cif", "rna", model_id=0)

coords = load_cif("data/biomole/4QG3.cif", "rna", mode="rna_only", model_id=0)
print("Number of atoms:", coords.shape[0])

barcode_from_point_cloud(coords, 10, 3, separate_h0_h1=False, save_fig="figs/biomole/rna_barcode012.png", dpi=500)