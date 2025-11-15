# import numpy as np
# import gudhi as gd
# import matplotlib.pyplot as plt

# # --- 1) Load your saved field ---
# eta = 0.4

# data = np.load(f"data/multires/field_eta{eta}.npz")
# F = data["F"].astype(float)               # shape (ny, nx); scalar field values
# # If you want SUPERLEVEL sets (high values appear first), invert:
# use_superlevel = False

# F = 1 - F / F.max()   # 线性归一化到 [0, 1]
# F = -F if use_superlevel else F

# # --- 2) Build a cubical complex ---
# # Gudhi expects the top-dimensional cell values (the scalar field) as a 2D array
# # (You can also pass a flattened 1D array plus `dimensions=[ny, nx]`.)
# cc = gd.CubicalComplex(top_dimensional_cells=F)

# # --- 3) Compute persistence ---
# # homology_coeff_field=2 (Z2) is standard; tweak min_persistence to denoise barcodes visually.
# diag = cc.persistence(homology_coeff_field=2, min_persistence=0.0)

# # --- 4) Inspect intervals by dimension ---
# H0 = cc.persistence_intervals_in_dimension(0)   # connected components
# H1 = cc.persistence_intervals_in_dimension(1)   # loops in 2D
# # (In 2D fields you typically care about H0 and H1.)

# print(f"H0 intervals: {H0.shape[0]}")
# print(f"H1 intervals: {H1.shape[0]}")

# # --- 5) Plot barcode and diagram ---
# fig, ax = plt.subplots(1, 2, figsize=(10, 4))
# gd.plot_persistence_barcode(diag, axes=ax[0])
# ax[0].set_title("Barcode")

# gd.plot_persistence_diagram(diag, axes=ax[1])
# ax[1].set_title("Persistence diagram")
# plt.tight_layout()
# plt.show()

# # --- 6) (Optional) Separate plots for H0 and H1 barcodes ---
# fig, axes = plt.subplots(1, 2, figsize=(12, 4))
# gd.plot_persistence_barcode(H0, axes=axes[0])
# axes[0].set_title(r"$\beta_0$ (Connected Components)")
# gd.plot_persistence_barcode(H1, axes=axes[1])
# axes[1].set_title(r"$\beta_1$ (Holes)")
# plt.tight_layout()
# plt.show()

import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt


def plot_persistence_from_field(
    data_dir: str,
    eta: float,
    use_superlevel: bool = False,
    separate_h0_h1: bool = False,
    save_fig: str | None = None,
    dpi: int = 300
) -> None:
    """
    Compute and plot persistence (barcode and diagram) from a saved 2D scalar field.

    Parameters
    ----------
    eta : float
        Resolution parameter used when saving the field (e.g., 0.2, 0.4, ...).
    use_superlevel : bool, optional
        If True, compute persistence of superlevel sets (high values appear first).
        If False, use sublevel filtration (default).
    separate_h0_h1 : bool, optional
        If True, plot separate barcodes for H0 (connected components) and H1 (holes).
        Otherwise, plot the combined barcode and persistence diagram.
    data_dir : str, optional
        Directory containing the saved `.npz` field files.
    """
    # --- 1) Load saved field ---
    path = f"{data_dir}/field_eta{eta}.npz"
    data = np.load(path)
    F = data["F"].astype(float)

    # --- 2) Normalize and optionally invert ---
    F = 1 - F / F.max()
    F = -F if use_superlevel else F
    #F_norm = np.clip(F_norm, 0.0, 1.0)

    # --- 3) Build cubical complex ---
    cc = gd.CubicalComplex(top_dimensional_cells=F)

    # --- 4) Compute persistence ---
    diag = cc.persistence(homology_coeff_field=2, min_persistence=0.0)

    # --- 5) Extract Betti intervals ---
    H0 = cc.persistence_intervals_in_dimension(0)
    H1 = cc.persistence_intervals_in_dimension(1)

    print(f"η = {eta:.2f}")
    print(f"  H₀ intervals: {H0.shape[0]}")
    print(f"  H₁ intervals: {H1.shape[0]}")

    # --- 6) Plot Barcode ---
    if separate_h0_h1:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        gd.plot_persistence_barcode(H0, axes=axes[0])
        axes[0].set_title(r"$\beta_0$ (Connected Components)")
        axes[0].set_xlabel("Filtration value")

        gd.plot_persistence_barcode(H1, axes=axes[1])
        axes[1].set_title(r"$\beta_1$ (Holes)")
        axes[1].set_xlabel("Filtration value")

    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        gd.plot_persistence_barcode(diag, axes=ax)
        ax.set_title("Persistence Barcode")
        ax.set_xlabel("Filtration value")
        
    fig.tight_layout()
    
    if save_fig:
        plt.savefig(save_fig, dpi=dpi)
    plt.show()

if __name__ == "__main__":
    # Example usage
    etas = [3.0, 4.0, 10.0, 30.0]
    for eta in etas:
        plot_persistence_from_field(
            data_dir="data/multires",
            eta=eta,
            use_superlevel=False,
            separate_h0_h1=False,
            save_fig=f"figs/hexagonal_frac/res{eta:.1f}_bar.png",
            dpi=500
        )
