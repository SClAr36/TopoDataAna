from Bio.PDB import MMCIFParser
import numpy as np

RNA_RESNAMES = {"A", "C", "G", "U"}

weight_dict = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8, 'S': 16, 'P': 15,
    'F': 9, 'CL': 17, 'BR': 35, 'I': 53,
}

def count_models(filename: str, structure_id: str = "X"):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, filename)
    
    models = list(structure.get_models())
    print(f"Number of models: {len(models)}")
    print("Model IDs:", [m.id for m in models])

    return models

def count_atoms_in_model(filename: str, structure_id: str = "X", model_id: int = 0):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, filename)

    # 获取所有 model（Bio.PDB 内部 id 可能从 0 或 1 开始，不假设索引）
    models = list(structure.get_models())

    if model_id < 0 or model_id > (len(models) - 1):
        raise ValueError(f"model_id must be between 0 and {len(models)-1}")

    model = models[model_id]

    atom_count = sum(1 for _ in model.get_atoms())
    print(f"Model {model_id} has {atom_count} atoms.")

    return atom_count


def load_cif(
    filename: str,
    structure_id: str,
    mode: str = "all",
    model_id: int = 0,
    output: str = "coords",  # "coords", "elements", "weights"
):
    """
    Unified loader returning (coords, second_output):
    
      output == "coords"
          return coords

      output == "coords_elements"
          return (coords, elements)

      output == "coords_weights"
          return (coords, weights)

    coords:  (N, 3) float32 array
    elements: list[str] of length N
    weights:  (N,) float32 array mapped by weight_dict
    """

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure(structure_id, filename)
    model = structure[model_id]

    coords = []
    elements = []

    for atom in model.get_atoms():
        name = atom.get_name()
        elem = atom.element.strip().upper()

        if mode == "all":
            coords.append(atom.get_coord())
            elements.append(elem)

        elif mode == "heavy":
            if not name.startswith("H"):
                coords.append(atom.get_coord())
                elements.append(elem)

        elif mode == "P":
            if elem == "P":  # 用 elem 判断更稳
                coords.append(atom.get_coord())
                elements.append(elem)

        elif mode == "rna_only":
            res = atom.get_parent()
            resname = res.get_resname().strip()

            if resname in RNA_RESNAMES:
                coords.append(atom.get_coord())
                elements.append(elem)

        else:
            raise ValueError(f"Unknown mode: {mode}")

    coords_arr = np.array(coords, dtype=np.float32)

    # --------- 统一输出模式 ----------
    if output == "coords":
        return coords_arr

    elif output == "elements":
        return coords_arr, elements

    elif output == "weights":
        weights = np.array(
            [weight_dict.get(e, 0.0) for e in elements],  # 未知元素给0，防止报错
            dtype=np.float32
        )
        return coords_arr, weights

    else:
        raise ValueError(f"Unknown output option: {output}")

def measure_molecule_size(coords: np.array) -> None:
    """
    Measure the spatial dimensions of a molecule loaded from a CIF file.

    Parameters
    ----------
    filepath : str
        Path to the .cif file.
    structure_id : str
        Arbitrary ID for loading the structure.
    mode : str
        Mode for loading atoms (e.g., "RNA", "all", etc.)
    """
    print(coords.shape)  # 预期 ~ (1723, 3)
    
    mins = coords.min(axis=0)  # [xmin, ymin, zmin]
    maxs = coords.max(axis=0)  # [xmax, ymax, zmax]
    Lx, Ly, Lz = maxs - mins
    
    print("x range:", mins[0], "->", maxs[0], "Å, length =", Lx)
    print("y range:", mins[1], "->", maxs[1], "Å, length =", Ly)
    print("z range:", mins[2], "->", maxs[2], "Å, length =", Lz)
    
    return Lx, Ly, Lz

def summarize_elements(elements):
    """
    Print unique atom types and counts.
    elements: list[str] length N
    """
    unique = sorted(set(elements))
    print("Atom types:", unique)

    # Count each element
    from collections import Counter
    cnt = Counter(elements)
    for e, n in cnt.items():
        print(f"  {e}: {n}")
    
    return cnt
