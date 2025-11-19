import numpy as np

from utils.barcode_utils import barcode_from_field
from utils.density_map import rigidity_density_map_3d_numba, normalize_density_map

weight_dict = {'C':6, 'N':7, 'O':8, 'S':16, 'P':15, 'F':9, 'CL':17, 'BR':35, 'I':53,}
etas = [1.0, 2.0, 6.0]

penta_xlim = (40, 150)
penta_ylim = (-56, 54)
penta_zlim = (100, 200)
hexa_xlim = (-112, 130)
hexa_ylim = (-180, 40)
hexa_zlim = (74, 200)


pentagon = np.load("data/biomole/virus_pentagon.npz", allow_pickle=True)
penta_coords = pentagon["coords"]        # (5705, 3)
penta_elem = pentagon["elements"]    # (5705,)

hexagon = np.load("data/biomole/virus_hexagon.npz", allow_pickle=True)
hexa_coords = hexagon["coords"]
hexa_elem = hexagon["elements"]

penta_weights = np.array([weight_dict[el.upper()] for el in penta_elem], dtype=float)
hexa_weights = np.array([weight_dict[el.upper()] for el in hexa_elem], dtype=float)

def print_shape_and_bounds(filename):
    data = np.load(filename, allow_pickle=True)
    coords = data["coords"]

    xyz_min = coords.min(axis=0)
    xyz_max = coords.max(axis=0)
    span = xyz_max - xyz_min

    print(f"\n>>> {filename}")
    print(f"Number of atoms: {coords.shape[0]}")
    print(f"X range: {xyz_min[0]:.3f} → {xyz_max[0]:.3f}  (span = {span[0]:.3f})")
    print(f"Y range: {xyz_min[1]:.3f} → {xyz_max[1]:.3f}  (span = {span[1]:.3f})")
    print(f"Z range: {xyz_min[2]:.3f} → {xyz_max[2]:.3f}  (span = {span[2]:.3f})")


# # 调用
# print_shape_and_bounds("data/biomole/virus_pentagon.npz")
# print_shape_and_bounds("data/biomole/virus_hexagon.npz")
for eta in etas:
    # pentagon
    ps, py, pz, rho = rigidity_density_map_3d_numba(penta_coords, xlim=penta_xlim, ylim=penta_ylim, zlim=penta_zlim, grid_spacing=0.6, eta=eta)
    
    np.savez(
        f"data/biomole/virus_penta_eta{eta}.npz",
        xs=ps,
        ys=py,
        zs=pz,
        rho=rho,
    )
    # rho = np.load(f"data/biomole/virus_penta_eta{eta}.npz")['rho']
    rho_norm = normalize_density_map(rho)
    barcode_from_field(rho_norm, separate_by_dim=True, min_bar_length=0.05, save_fig=f"figs/biomole/virus_penta_eta{eta}_sep.png", dpi=500)

    # hexagon
    hs, hy, hz, psi = rigidity_density_map_3d_numba(hexa_coords, xlim=hexa_xlim, ylim=hexa_ylim, zlim=hexa_zlim, grid_spacing=0.6, eta=eta)
    
    np.savez(
        f"data/biomole/virus_hexa_eta{eta}.npz",
        xs=hs,
        ys=hy,
        zs=hz,
        rho=psi,
    )
    # psi = np.load(f"data/biomole/virus_hexa_eta{eta}.npz")['rho']
    psi_norm = normalize_density_map(psi)
    barcode_from_field(psi_norm, separate_by_dim=True, min_bar_length=0.05, save_fig=f"figs/biomole/virus_hexa_eta{eta}_sep.png", dpi=500)
