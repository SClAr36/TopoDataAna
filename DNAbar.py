import numpy as np
import gudhi
import matplotlib.pyplot as plt
from Bio.PDB import MMCIFParser

def barcode_from_point_cloud(points,
                             max_edge_len: float,
                             max_dim: int =3,
                             min_pers: float = 0.00,
                             plt_barcode: bool = True,
                             save_fig: str | None = None,
                             dpi: int = 500) -> None:
    
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

def load_cif(filename: str, structure_id: str, mode: str = "all") -> np.ndarray:

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, filename)

    if mode == "all":
        coords = [atom.get_coord() for atom in structure.get_atoms()]
        return np.array(coords)

    if mode == "heavy":
        coords = []
        for atom in structure.get_atoms():
            if not atom.get_name().startswith("H"):  # 非氢原子
                coords.append(atom.get_coord())
        return np.array(coords)

    if mode == "P":
        coords = []
        for atom in structure.get_atoms():
            if atom.get_name() == "P":
                coords.append(atom.get_coord())
        return np.array(coords)


def load_dna(mode: str) -> np.ndarray:
    coords = load_cif("data/biomole/2M54.cif", "dna", mode=mode)
    return coords

coords_all = load_dna("all")
coords_heavy = load_dna("heavy")
coords_P = load_dna("P")

barcode_from_point_cloud(coords_P, 10, 2, save_fig="heavy.png")
