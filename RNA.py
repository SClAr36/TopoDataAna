import numpy as np

from utils.load_cif import load_cif
from utils.barcode_utils import barcode_from_point_cloud

coords = load_cif("data/biomole/4QG3.cif", "rna", mode="rna_only", model_id=0)

barcode_from_point_cloud(coords, 10, 3, separate_h0_h1=False, save_fig="figs/biomole/rna_barcode.png", dpi=500)