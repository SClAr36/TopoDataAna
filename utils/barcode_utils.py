import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from pathlib import Path

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

def barcode_from_field(
    field: np.ndarray,
    #use_superlevel: bool = False,
    homology_coeff_field=2,
    separate_h0_h1: bool = False,
    save_fig: str | None = None,
    dpi: int = 300,
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


    # --- 3) Build cubical complex ---
    cc = gd.CubicalComplex(top_dimensional_cells=field)

    # --- 4) Compute persistence ---
    diag = cc.persistence(homology_coeff_field=homology_coeff_field, min_persistence=0.0)

    # --- 5) Extract Betti intervals ---
    H0 = cc.persistence_intervals_in_dimension(0)
    H1 = cc.persistence_intervals_in_dimension(1)

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
        save_dir = Path(save_fig)
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_fig, dpi=dpi)
    plt.show()





