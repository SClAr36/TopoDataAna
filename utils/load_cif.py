from Bio.PDB import MMCIFParser
import numpy as np

RNA_RESNAMES = {"A", "C", "G", "U"}   # 只保留 RNA

def load_cif(filename: str, structure_id: str, mode: str = "all", model_id: int = 1) -> np.ndarray:
    """
    Load atom coordinates from an mmCIF file.

    Parameters
    ----------
    filename : str
        Path to the .cif file.
    structure_id : str
        Arbitrary ID for loading the structure.
    mode : str
        "all"  - all atoms (including hydrogens)
        "heavy" - heavy atoms only (exclude H)
        "P"     - phosphorus atoms only
    model_id : int
        Which NMR model to use (default = 1)

    Returns
    -------
    coords : np.ndarray
        Array of shape (N, 3) containing Cartesian coordinates.
    """

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, filename)

    # ---- IMPORTANT: only select one model ----
    model = structure[model_id]   # works if model numbering starts from 1

    coords = []

    for atom in model.get_atoms():
        name = atom.get_name()

        if mode == "all":
            coords.append(atom.get_coord())

        elif mode == "heavy":
            if not name.startswith("H"):
                coords.append(atom.get_coord())

        elif mode == "P":
            if name == "P":
                coords.append(atom.get_coord())
        
        elif mode == "rna_only":
            for atom in model.get_atoms():
                res = atom.get_parent()
                resname = res.get_resname().strip()
        
                # 只保留 RNA，排除蛋白质和离子
                if resname not in RNA_RESNAMES:
                    continue
        
                coords.append(atom.get_coord())

        else:
            raise ValueError(f"Unknown mode: {mode}")

    return np.array(coords)
