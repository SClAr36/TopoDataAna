import numpy as np

from utils.barcode_utils import barcode_from_field
from utils.density_map import rigidity_density_map_3d_numba, normalize_density_map

weight_dict = {'C':6, 'N':7, 'O':8, 'S':16, 'P':15, 'F':9, 'CL':17, 'BR':35, 'I':53,}
etas = [1.0, 2.0, 6.0]


penta_xlim = (40, 150)
penta_ylim = (-56, 54)
penta_zlim = (100, 200)
hexa_xlim = (-62, 62)
hexa_ylim = (-64, 64)
hexa_zlim = (140, 215)

# load pentagon
pentagon = np.load("data/biomole/virus_pentagon.npz", allow_pickle=True)
penta_coords = pentagon["coords"]
penta_elem = pentagon["elements"]

# load hexagon 
data = np.load("data/biomole/selected_BCDhexagon.npz", allow_pickle=True)

# 读取数据
B1 = data["B1_coords"]; B1_e = data["B1_elems"]
B6 = data["B6_coords"]; B6_e = data["B6_elems"]
C1 = data["C1_coords"]; C1_e = data["C1_elems"]
C6 = data["C6_coords"]; C6_e = data["C6_elems"]
D1 = data["D1_coords"]; D1_e = data["D1_elems"]
D6 = data["D6_coords"]; D6_e = data["D6_elems"]

hexa_coords = np.concatenate([B1, B6, C1, C6, D1, D6], axis=0)
hexa_elem  = np.concatenate([B1_e, B6_e, C1_e, C6_e, D1_e, D6_e], axis=0)

print("[INFO] merged coords:", hexa_coords.shape)
print("[INFO] merged elems :", hexa_elem.shape)

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


# 调用
print_shape_and_bounds("data/biomole/virus_pentagon.npz")
xmin, ymin, zmin = hexa_coords.min(axis=0)
xmax, ymax, zmax = hexa_coords.max(axis=0)

print(f"X range: {xmin:.3f} → {xmax:.3f} (span = {(xmax - xmin):.3f})")
print(f"Y range: {ymin:.3f} → {ymax:.3f} (span = {(ymax - ymin):.3f})")
print(f"Z range: {zmin:.3f} → {zmax:.3f} (span = {(zmax - zmin):.3f})")

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
