import numpy as np

from utils.load_cif import load_cif, summarize_elements, measure_molecule_size
from utils.density_map import rigidity_density_map_3d_numba, normalize_density_map
from utils.barcode_utils import barcode_from_field

weight_dict = {'C':6, 'N':7, 'O':8, 'S':16, 'P':15, 'F':9, 'CL':17, 'BR':35, 'I':53,}
xlim = (-40, 25)
ylim = (-25, 40)
zlim = (-35, 25)
etas = [0.3, 0.5, 0.7, 1.0, 2.0, 4.0]

coords, weights = load_cif("data/biomole/4QG3.cif", "rna", mode="rna_only", output="weights")

for eta in etas:
    xs, ys, zs, rho = rigidity_density_map_3d_numba(points=coords, xlim=xlim, ylim=ylim, zlim=zlim, grid_spacing=0.3, eta=eta, kappa=2.0, weights=weights)

    # 存成一个 npz 文件（不压缩）
    np.savez(
        f"data/biomole/4QG3_density_eta{eta}.npz",
        xs=xs,
        ys=ys,
        zs=zs,
        rho=rho,
    )
    #rho = np.load(f"data/biomole/4QG3_density_eta{eta}.npz")['rho']
    
    F = normalize_density_map(rho, clip=True, clip_min=0.0, clip_max=1.0)
    barcode_from_field(field=F, homology_coeff_field=2, min_bar_length=0.05, save_fig=f"figs/biomole/rna_eta{eta}_barcode(cut).png", dpi=500)





