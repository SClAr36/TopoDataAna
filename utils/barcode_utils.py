import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from pathlib import Path

#TODO: cut short bars

#TODO: add separate version (and change color of bars)
def barcode_from_point_cloud(points,
                             max_edge_len: float,
                             max_dim: int =3,
                             min_pers: float = 0.00,
                             separate_h0_h1: bool = False,
                             plt_barcode: bool = True,
                             save_fig: str | None = None,
                             dpi: int = 300) -> None:
    
    # generate rips complex
    rips_complex = gd.RipsComplex(points=points, max_edge_length=max_edge_len)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
    diag = simplex_tree.persistence(min_persistence=min_pers)
    
    # if separate_h0_h1:
    #     H0 = simplex_tree.persistence_intervals_in_dimension(0)
    #     H1 = simplex_tree.persistence_intervals_in_dimension(1)
    
    #     print(f"  H₀ intervals: {H0.shape[0]}")
    #     print(f"  H₁ intervals: {H1.shape[0]}")

    #     fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    #     gd.plot_persistence_barcode(H0, axes=axes[0])
    #     axes[0].set_title(r"$\beta_0$ (Connected Components)")
    #     axes[0].set_xlabel("Filtration value")

    #     gd.plot_persistence_barcode(H1, axes=axes[1])
    #     axes[1].set_title(r"$\beta_1$ (Holes)")
    #     axes[1].set_xlabel("Filtration value")

    # plot barcode
    gd.plot_persistence_barcode(diag)
    if save_fig:
        plt.savefig(save_fig, dpi=dpi)
    if plt_barcode:
        plt.show()


#TODO: add to 3d separate

# def barcode_from_field(
#     field: np.ndarray,
#     #use_superlevel: bool = False,
#     homology_coeff_field=2,
#     separate_h0_h1: bool = False,
#     save_fig: str | None = None,
#     dpi: int = 300,
# ) -> None:
#     """
#     Compute and plot persistence (barcode and diagram) from a saved 2D scalar field.

#     Parameters
#     ----------
#     eta : float
#         Resolution parameter used when saving the field (e.g., 0.2, 0.4, ...).
#     use_superlevel : bool, optional
#         If True, compute persistence of superlevel sets (high values appear first).
#         If False, use sublevel filtration (default).
#     separate_h0_h1 : bool, optional
#         If True, plot separate barcodes for H0 (connected components) and H1 (holes).
#         Otherwise, plot the combined barcode and persistence diagram.
#     data_dir : str, optional
#         Directory containing the saved `.npz` field files.
#     """


#     # --- 3) Build cubical complex ---
#     cc = gd.CubicalComplex(top_dimensional_cells=field)

#     # --- 4) Compute persistence ---
#     diag = cc.persistence(homology_coeff_field=homology_coeff_field, min_persistence=0.0)

#     # --- 5) Extract Betti intervals ---
#     H0 = cc.persistence_intervals_in_dimension(0)
#     H1 = cc.persistence_intervals_in_dimension(1)

#     print(f"  H₀ intervals: {H0.shape[0]}")
#     print(f"  H₁ intervals: {H1.shape[0]}")

#     # --- 6) Plot Barcode ---
#     if separate_h0_h1:
#         fig, axes = plt.subplots(1, 2, figsize=(12, 4))

#         gd.plot_persistence_barcode(H0, axes=axes[0])
#         axes[0].set_title(r"$\beta_0$ (Connected Components)")
#         axes[0].set_xlabel("Filtration value")

#         gd.plot_persistence_barcode(H1, axes=axes[1])
#         axes[1].set_title(r"$\beta_1$ (Holes)")
#         axes[1].set_xlabel("Filtration value")

#     else:
#         fig, ax = plt.subplots(figsize=(6, 4))
#         gd.plot_persistence_barcode(diag, axes=ax)
#         ax.set_title("Persistence Barcode")
#         ax.set_xlabel("Filtration value")
        
#     fig.tight_layout()
    
#     if save_fig:
#         save_dir = Path(save_fig)
#         save_dir.parent.mkdir(parents=True, exist_ok=True)
#         plt.savefig(save_fig, dpi=dpi)
#     plt.show()

def barcode_from_field(
    field: np.ndarray,
    homology_coeff_field=2,
    separate_by_dim: bool = False,   # renamed parameter
    min_bar_length: float = 0.0,
    save_fig: str | None = None,
    dpi: int = 300,
) -> None:
    """
    Compute and plot persistence barcodes from a 2D or 3D scalar field.
    """

    # 1) Build cubical complex
    cc = gd.CubicalComplex(top_dimensional_cells=field)

    # 2) Compute all persistence pairs
    diag_raw = cc.persistence(
        homology_coeff_field=homology_coeff_field,
        min_persistence=0.0
    )

    # 3) Extract intervals by dimension
    H0 = cc.persistence_intervals_in_dimension(0)
    H1 = cc.persistence_intervals_in_dimension(1)
    try:
        H2 = cc.persistence_intervals_in_dimension(2)
    except Exception:
        H2 = np.empty((0, 2))

    print(f"H0: {H0.shape[0]}, H1: {H1.shape[0]}, H2: {H2.shape[0]}")

    # 4) Filter out short bars
    if min_bar_length > 0:
        H0 = H0[(H0[:, 1] - H0[:, 0]) >= min_bar_length]
        H1 = H1[(H1[:, 1] - H1[:, 0]) >= min_bar_length]
        H2 = H2[(H2[:, 1] - H2[:, 0]) >= min_bar_length]

        diag = [
            pair for pair in diag_raw
            if (pair[1][1] - pair[1][0]) >= min_bar_length
        ]
    else:
        diag = diag_raw

    # 5) Plot
    if separate_by_dim:
        dims_available = [H0, H1] + ([H2] if H2.shape[0] > 0 else [])
        ncols = len(dims_available)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))

        idx = 0
        if H0.shape[0] > 0:
            gd.plot_persistence_barcode(H0, axes=axes[idx])
            axes[idx].set_title("H0")
            idx += 1

        if H1.shape[0] > 0:
            gd.plot_persistence_barcode(H1, axes=axes[idx])
            axes[idx].set_title("H1")
            idx += 1

        if H2.shape[0] > 0:
            gd.plot_persistence_barcode(H2, axes=axes[idx])
            axes[idx].set_title("H2")

    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        gd.plot_persistence_barcode(diag, axes=ax)
        ax.set_title("Persistence Barcode")

    # 6) Save
    fig.tight_layout()
    if save_fig:
        Path(save_fig).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig, dpi=dpi)

    plt.show()




