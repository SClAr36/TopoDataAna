# import numpy as np
# from Bio.PDB import MMCIFParser
# from Bio.PDB.MMCIF2Dict import MMCIF2Dict   # ← 正确 import 路径！


# def load_basic_chain_coords(filename: str) -> np.ndarray:
#     """
#     Load the asymmetric-unit protein coordinates from 1DYL,
#     and return the coordinates of the FIRST polymer chain as the template block.

#     Returns
#     -------
#     coords : ndarray of shape (N_atoms, 3)
#     """
#     parser = MMCIFParser(QUIET=True)
#     structure = parser.get_structure("1DYL", filename)

#     # Take model 0
#     model = next(structure.get_models())

#     # Select first polymer chain (skip water/ions)
#     for chain in model:
#         if chain.id.strip() not in ("", "W"):  # remove water/empty id
#             coords = np.array([atom.coord for atom in chain.get_atoms()], dtype=float)
#             return coords

#     raise RuntimeError("No valid polymer chain found in CIF file.")


# def load_sym_ops(filename: str):
#     """
#     Parse symmetry operations (rotation + translation) from mmCIF.
#     Only keep operations whose IDs are integers.
#     Returns dict: op_id(int) -> (R, t)
#     """
#     cif = MMCIF2Dict(filename)

#     ids = cif["_pdbx_struct_oper_list.id"]

#     m11 = cif["_pdbx_struct_oper_list.matrix[1][1]"]
#     m12 = cif["_pdbx_struct_oper_list.matrix[1][2]"]
#     m13 = cif["_pdbx_struct_oper_list.matrix[1][3]"]
#     m21 = cif["_pdbx_struct_oper_list.matrix[2][1]"]
#     m22 = cif["_pdbx_struct_oper_list.matrix[2][2]"]
#     m23 = cif["_pdbx_struct_oper_list.matrix[2][3]"]
#     m31 = cif["_pdbx_struct_oper_list.matrix[3][1]"]
#     m32 = cif["_pdbx_struct_oper_list.matrix[3][2]"]
#     m33 = cif["_pdbx_struct_oper_list.matrix[3][3]"]

#     v1 = cif["_pdbx_struct_oper_list.vector[1]"]
#     v2 = cif["_pdbx_struct_oper_list.vector[2]"]
#     v3 = cif["_pdbx_struct_oper_list.vector[3]"]

#     ops = {}

#     for i, op_id in enumerate(ids):
#         # skip non-numeric IDs
#         try:
#             op_id_int = int(op_id)
#         except ValueError:
#             continue

#         R = np.array([
#             [float(m11[i]), float(m12[i]), float(m13[i])],
#             [float(m21[i]), float(m22[i]), float(m23[i])],
#             [float(m31[i]), float(m32[i]), float(m33[i])],
#         ])
#         t = np.array([float(v1[i]), float(v2[i]), float(v3[i])])

#         ops[op_id_int] = (R, t)

#     return ops


# def build_complex(X: np.ndarray, op_ids, sym_ops) -> np.ndarray:
#     """
#     Apply multiple symmetry operations to the template block X.
#     Return concatenated atom coordinates.
#     """
#     blocks = []
#     for op_id in op_ids:
#         R, t = sym_ops[op_id]
#         X_new = X @ R.T + t
#         blocks.append(X_new)
#     return np.concatenate(blocks, axis=0)


# def extract_pentagon_hexagon(filename: str):
#     """
#     High-level function:
#     - load template chain
#     - load symmetry ops
#     - build pentagon and hexagon complexes
#     """

#     # 1) Load asymmetric-unit chain
#     X = load_basic_chain_coords(filename)
#     print(f"[INFO] Template chain loaded: {X.shape[0]} atoms")

#     # 2) Load symmetry operations
#     sym_ops = load_sym_ops(filename)
#     print(f"[INFO] Symmetry operations loaded: {len(sym_ops)} ops")

#     # Verified correct op sets for 1DYL capsid
#     pentagon_ops = [1, 2, 3, 4, 5]
#     hexagon_ops  = [1, 2, 6, 10, 23, 24]

#     # 3) Build complexes
#     pentagon = build_complex(X, pentagon_ops, sym_ops)
#     hexagon  = build_complex(X, hexagon_ops , sym_ops)

#     print(f"[INFO] Pentagon shape = {pentagon.shape}")
#     print(f"[INFO] Hexagon shape  = {hexagon.shape}")

#     return pentagon, hexagon


# if __name__ == "__main__":
#     penta, hexa = extract_pentagon_hexagon("data/biomole/1DYL.cif")

#     # Optional: save for later persistent homology / FRI
#     np.save("data/biomole/virus_pentagon.npy", penta)
#     np.save("data/biomole/virus_hexagon.npy", hexa)

#     print("[DONE] Saved pentagon.npy and hexagon.npy")

import numpy as np
from collections import Counter
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


# =============================================================
# Helper 1 — Remove duplicate atoms while keeping element info
# =============================================================
def remove_duplicate_atoms(coords, elements, tol=1e-6):
    """
    Remove duplicate atoms (geometric duplicates) and
    keep element mapping aligned.
    """
    rounded = np.round(coords / tol).astype(np.int64)

    # find unique rows
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    unique_idx = sorted(unique_idx)

    coords_new = coords[unique_idx]
    elements_new = elements[unique_idx]

    return coords_new, elements_new


# =============================================================
# Helper 2 — Load template chain (coords + element)
# =============================================================
def load_basic_chain(filename: str):
    """
    Return (coords, elements) for the first polymer chain.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("1DYL", filename)
    model = next(structure.get_models())

    for chain in model:
        cid = chain.id.strip()
        if cid not in ("", "W"):  # skip water / empty
            coords = []
            elements = []
            for atom in chain.get_atoms():
                coords.append(atom.coord)
                elements.append(atom.element)

            coords = np.array(coords, float)
            elements = np.array(elements, dtype=object)

            return coords, elements

    raise RuntimeError("No valid polymer chain found.")


# =============================================================
# Helper 3 — Load symmetry operations (numeric IDs only)
# =============================================================
def load_sym_ops(filename: str):
    cif = MMCIF2Dict(filename)

    ids = cif["_pdbx_struct_oper_list.id"]

    m11 = cif["_pdbx_struct_oper_list.matrix[1][1]"]
    m12 = cif["_pdbx_struct_oper_list.matrix[1][2]"]
    m13 = cif["_pdbx_struct_oper_list.matrix[1][3]"]
    m21 = cif["_pdbx_struct_oper_list.matrix[2][1]"]
    m22 = cif["_pdbx_struct_oper_list.matrix[2][2]"]
    m23 = cif["_pdbx_struct_oper_list.matrix[2][3]"]
    m31 = cif["_pdbx_struct_oper_list.matrix[3][1]"]
    m32 = cif["_pdbx_struct_oper_list.matrix[3][2]"]
    m33 = cif["_pdbx_struct_oper_list.matrix[3][3]"]

    v1 = cif["_pdbx_struct_oper_list.vector[1]"]
    v2 = cif["_pdbx_struct_oper_list.vector[2]"]
    v3 = cif["_pdbx_struct_oper_list.vector[3]"]

    ops = {}

    for i, op_id in enumerate(ids):
        # skip non-numeric ops
        try:
            op_id_int = int(op_id)
        except ValueError:
            continue

        R = np.array([
            [float(m11[i]), float(m12[i]), float(m13[i])],
            [float(m21[i]), float(m22[i]), float(m23[i])],
            [float(m31[i]), float(m32[i]), float(m33[i])],
        ])
        t = np.array([float(v1[i]), float(v2[i]), float(v3[i])])

        ops[op_id_int] = (R, t)

    return ops


# =============================================================
# Helper 4 — Build complex (coords + elements)
# =============================================================
def build_complex(X, elements, op_ids, sym_ops):
    coords_blocks = []
    elem_blocks = []

    for op_id in op_ids:
        R, t = sym_ops[op_id]
        coords_blocks.append(X @ R.T + t)
        elem_blocks.append(elements)

    coords = np.concatenate(coords_blocks, axis=0)
    elems = np.concatenate(elem_blocks, axis=0)

    return coords, elems


# =============================================================
# Final — Extract pentamer and hexamer with elements
# =============================================================
def extract_pentagon_hexagon(filename: str):
    # 1) Load template chain
    X, X_elem = load_basic_chain(filename)
    print(f"[INFO] Template chain loaded: {X.shape[0]} atoms")

    # 2) Load symmetry ops
    sym_ops = load_sym_ops(filename)
    print(f"[INFO] Symmetry operations loaded: {len(sym_ops)} ops")

    # Verified correct op sets for 1DYL capsid
    pentagon_ops = [1, 2, 3, 4, 5]
    hexagon_ops  = [1, 2, 6, 10, 23, 24]

    # 3) Build raw complexes
    penta_raw, penta_elem_raw = build_complex(X, X_elem, pentagon_ops, sym_ops)
    hexa_raw, hexa_elem_raw   = build_complex(X, X_elem, hexagon_ops , sym_ops)

    # 4) Remove duplicates
    penta, penta_elem = remove_duplicate_atoms(penta_raw, penta_elem_raw)
    hexa, hexa_elem   = remove_duplicate_atoms(hexa_raw , hexa_elem_raw)

    print(f"[INFO] Pentagon shape = {penta.shape}")
    print(f"[INFO] Hexagon shape  = {hexa.shape}")

    # 5) Print element statistics
    print("\n===== Pentagon Atom Composition =====")
    count_p = Counter(penta_elem)
    for e, n in count_p.items():
        print(f"  {e}: {n}")
    print(f"Total: {len(penta_elem)} atoms")

    print("\n===== Hexagon Atom Composition =====")
    count_h = Counter(hexa_elem)
    for e, n in count_h.items():
        print(f"  {e}: {n}")
    print(f"Total: {len(hexa_elem)} atoms\n")

    return (penta, penta_elem), (hexa, hexa_elem)


# =============================================================
# If run directly
# =============================================================
if __name__ == "__main__":
    (penta, penta_elem), (hexa, hexa_elem) = extract_pentagon_hexagon("data/biomole/1DYL.cif")

    np.savez("pentagon.npz", coords=penta, elements=penta_elem)
    np.savez("hexagon.npz", coords=hexa, elements=hexa_elem)

    print("[DONE] Saved pentagon.npz and hexagon.npz")
