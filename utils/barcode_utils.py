import numpy as np
import gudhi
import matplotlib.pyplot as plt

def barcode_from_point_cloud(points,
                             max_edge_len: float,
                             max_dim: int =3,
                             min_pers: float = 0.00,
                             plt_barcode: bool = True,
                             save_fig: str | None = None,
                             dpi: int = 300) -> None:
    
    # generate rips complex
    rips_complex = gudhi.RipsComplex(points=points, max_edge_length=max_edge_len)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dim)
    diag = simplex_tree.persistence(min_persistence=min_pers)
    # plot barcode
    gudhi.plot_persistence_barcode(diag)
    if save_fig:
        plt.savefig(save_fig, dpi=dpi)
    if plt_barcode:
        plt.show()

# def cubical_cplx_barcode(field: np.ndarray,
#                          homology_coeff_field: int = 2,
#                          min_pers: float = 0.0,
#                          use_superlevel: bool = False,
#                          plt_barcode: bool = True,
#                          save_fig: str | None = None,
#                          dpi: int = 300) -> None:





