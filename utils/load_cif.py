from Bio.PDB import MMCIFParser
import numpy as np

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
