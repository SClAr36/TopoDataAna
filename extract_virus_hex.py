import numpy as np
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


# =============================================================
# 1. 读取四条 polymer 链 A/B/C/D 的原子坐标+元素
# =============================================================
def load_chains_coords(cif_path: str):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("1DYL", cif_path)
    model = next(structure.get_models())

    chains = {}
    elems = {}

    for cid in ["A", "B", "C", "D"]:
        chains[cid] = []
        elems[cid] = []

    for chain in model:
        cid = chain.id.strip()
        if cid in chains:
            for atom in chain.get_atoms():
                res = atom.get_parent()
                # 只保留标准氨基酸残基
                if res.get_id()[0] == " ":
                    chains[cid].append(atom.coord)
                    elems[cid].append(atom.element)

    for cid in chains:
        chains[cid] = np.array(chains[cid], float)  # (n,3)
        elems[cid] = np.array(elems[cid], object)   # (n,)
        print(f"[INFO] Chain {cid}: {chains[cid].shape[0]} atoms")

    return chains, elems


# =============================================================
# 2. 读取 60 个对称操作
# =============================================================
def load_sym_ops(cif_path: str):
    cif = MMCIF2Dict(cif_path)
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
            continue

        R = np.array([
            [float(m11[i]), float(m12[i]), float(m13[i])],
            [float(m21[i]), float(m22[i]), float(m23[i])],
            [float(m31[i]), float(m32[i]), float(m33[i])]
        ], float)
        t = np.array([float(v1[i]), float(v2[i]), float(v3[i])], float)

        ops[op_id_int] = (R, t)

    print(f"[INFO] Loaded {len(ops)} symmetry operations.")
    return ops


# =============================================================
# 3. 构造完整 capsid + group labels + centers
# =============================================================
def build_full_capsid_with_labels(chains, elems, sym_ops):
    """
    返回:
      full_coords   : (273780,3)
      full_elems    : (273780,)
      full_labels   : (273780,2)   每个原子对应 (chain_idx, op_id)
      centers       : (240,3)      每个 group 的中心点
      center_labels : (240,2)      group key (chain_idx, op_id)
    """

    chain_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    op_ids_sorted = sorted(sym_ops.keys())

    all_coords = []
    all_elems = []
    all_labels = []

    center_points = []
    center_labels = []

    for cid in ["A", "B", "C", "D"]:
        cid_idx = chain_map[cid]
        base_coords = chains[cid]
        base_elems = elems[cid]

        for op_id in op_ids_sorted:
            R, t = sym_ops[op_id]

            coords_trsf = base_coords @ R.T + t

            # append all atoms
            all_coords.append(coords_trsf)
            all_elems.append(base_elems)

            # labels for each atom
            n_atoms = coords_trsf.shape[0]
            labels_block = np.column_stack([
                np.full(n_atoms, cid_idx, int),
                np.full(n_atoms, op_id, int)
            ])
            all_labels.append(labels_block)

            # compute center for this group
            center = coords_trsf.mean(axis=0)
            center_points.append(center)
            center_labels.append((cid_idx, op_id))

    # concatenate big arrays
    full_coords = np.concatenate(all_coords, axis=0)
    full_elems = np.concatenate(all_elems, axis=0)
    full_labels = np.concatenate(all_labels, axis=0)

    centers = np.array(center_points, float)
    center_labels = np.array(center_labels, int)

    print("[INFO] Full capsid constructed:")
    print("       coords:", full_coords.shape)
    print("       elems :", full_elems.shape)
    print("       labels:", full_labels.shape)
    print("       centers:", centers.shape)

    return full_coords, full_elems, full_labels, centers, center_labels

# =============================================================
# 4. 选择特定 chain + op_id 的原子
# =============================================================
def select_copy(full_coords, full_elems, full_labels, chain, op_id):
    chain_map = {"A":0,"B":1,"C":2,"D":3}
    cid_idx = chain_map[chain]
    mask = (full_labels[:,0]==cid_idx) & (full_labels[:,1]==op_id)
    return full_coords[mask], full_elems[mask]


# =============================================================
# 主程式
# =============================================================
if __name__ == "__main__":
    cif_path = "data/biomole/1DYL.cif"   # 改成你的路径

    # 1. 读 A/B/C/D 四链
    chains, elems = load_chains_coords(cif_path)

    # 2. 读 60 个对称操作
    sym_ops = load_sym_ops(cif_path)

    # 3. 构造完整 capsid + labels + centers
    full_coords, full_elems, full_labels, centers, center_labels = \
        build_full_capsid_with_labels(chains, elems, sym_ops)

    # 4. 保存完整 capsid
    np.savez(
        "data/biomole/capsid_full_with_labels.npz",
        coords=full_coords,
        elems=full_elems,
        labels=full_labels,
        centers=centers,
        center_labels=center_labels,
    )
    print("[DONE] saved capsid_full_with_labels.npz")

    # =============================================================
    # 额外：提取 B1, B6, C1, C6, D1, D6 的所有原子
    # =============================================================
    print("\n[INFO] Extracting B1, B6, C1, C6, D1, D6 ...")

    targets = [
        ("B", 1), ("B", 6),   ### 修改：原 B2 → B6
        ("C", 1), ("C", 6),   ### 修改：原 C2 → C6
        ("D", 1), ("D", 6),   ### 修改：原 D2 → D6
    ]

    extracted_coords = {}
    extracted_elems  = {}

    for chain, op_id in targets:
        coords_i, elems_i = select_copy(full_coords, full_elems, full_labels, chain, op_id)
        key = f"{chain}{op_id}"
        extracted_coords[key] = coords_i
        extracted_elems[key]  = elems_i
        print(f"[INFO] {key}: {coords_i.shape[0]} atoms")


    # =============================================================
    # 保存这 6 个 copy 的 coords + elems
    # =============================================================
    np.savez(
        "data/biomole/selected_BCD_1_6.npz",     ### 修改：文件名从 *_1_2 → *_1_6
        B1_coords = extracted_coords["B1"],
        B1_elems  = extracted_elems["B1"],
        B6_coords = extracted_coords["B6"],      ### 修改：原 B2 → B6
        B6_elems  = extracted_elems["B6"],       ### 修改

        C1_coords = extracted_coords["C1"],
        C1_elems  = extracted_elems["C1"],
        C6_coords = extracted_coords["C6"],      ### 修改
        C6_elems  = extracted_elems["C6"],       ### 修改

        D1_coords = extracted_coords["D1"],
        D1_elems  = extracted_elems["D1"],
        D6_coords = extracted_coords["D6"],      ### 修改
        D6_elems  = extracted_elems["D6"],       ### 修改
    )
    print("[DONE] saved selected_BCD_1_6.npz")
