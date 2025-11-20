import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


# =============================================================
# 1) 读取所有 polymer 链（A/B/C/D），返回所有原子坐标和元素
# =============================================================
def load_all_polymer_atoms(filename: str):
    """
    Return coords (N,3) and elements (N,) for ALL polymer chains in 1DYL,
    i.e. chains A/B/C/D, excluding waters/ligands.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("1DYL", filename)
    model = next(structure.get_models())

    coords = []
    elems = []

    for chain in model:
        cid = chain.id.strip()
        # 这里我们只要多肽链，跳过水/空链，如果你以后有别的需求再改
        if cid in ("A", "B", "C", "D"):
            for atom in chain.get_atoms():
                # parent id[0] == " " 表示标准残基（不是 HETATM / 水）
                res = atom.get_parent()
                if res.get_id()[0] == " ":
                    coords.append(atom.coord)
                    elems.append(atom.element)

    coords = np.array(coords, dtype=float)
    elems = np.array(elems, dtype=object)

    print(f"[INFO] Total polymer atoms in asymmetric unit = {coords.shape[0]}")
    return coords, elems


# =============================================================
# 2) 读取所有对称操作 (id 为数字的 1..60)
# =============================================================
def load_sym_ops(filename: str):
    """
    Return a dict: {op_id: (R, t)} for all numeric symmetry operations.
    """
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
        try:
            op_id_int = int(op_id)
        except ValueError:
            # 比如 "P" 这种非数字 ID，直接跳过
            continue

        R = np.array(
            [
                [float(m11[i]), float(m12[i]), float(m13[i])],
                [float(m21[i]), float(m22[i]), float(m23[i])],
                [float(m31[i]), float(m32[i]), float(m33[i])],
            ],
            dtype=float,
        )
        t = np.array([float(v1[i]), float(v2[i]), float(v3[i])], dtype=float)

        ops[op_id_int] = (R, t)

    print(f"[INFO] Loaded {len(ops)} symmetry operations.")
    return ops


# =============================================================
# 3) 构造完整 capsid：所有链 × 所有对称操作
# =============================================================
def build_full_capsid(filename: str):
    """
    Build full viral capsid for 1DYL:
    - Use all polymer chains in ASU (A/B/C/D)
    - Apply all numeric symmetry ops (1..60)
    Return:
        coords_full: (273780, 3)
        elems_full:  (273780,)
    """
    base_coords, base_elems = load_all_polymer_atoms(filename)  # (4563, 3), (4563,)
    sym_ops = load_sym_ops(filename)

    all_coords = []
    all_elems = []

    for op_id in sorted(sym_ops.keys()):
        R, t = sym_ops[op_id]
        transformed = base_coords @ R.T + t  # (4563, 3)
        all_coords.append(transformed)
        all_elems.append(base_elems)

    coords_full = np.concatenate(all_coords, axis=0)
    elems_full = np.concatenate(all_elems, axis=0)

    print(f"[INFO] Full capsid coords shape = {coords_full.shape}")
    print(f"[INFO] Full capsid elems  shape = {elems_full.shape}")

    return coords_full, elems_full


# =============================================================
# 4) 如果作为脚本直接运行：构造并保存
# =============================================================
if __name__ == "__main__":
    # 改成你自己的 cif 路径
    filename = "data/biomole/1DYL.cif"

    capsid_coords, capsid_elems = build_full_capsid(filename)

    # 保存成 npz，方便后续读
    np.savez("data/biomole/capsid_full.npz", coords=capsid_coords, elements=capsid_elems)
    print("[DONE] Saved capsid_full.npz")
