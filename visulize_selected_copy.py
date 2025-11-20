import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#6,7,32,58

# =============================================================
# 选择特定 chain + op_id 的原子
# =============================================================
def select_copy(full_coords, full_elems, full_labels, chain, op_id):
    chain_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    cid_idx = chain_map[chain]
    mask = (full_labels[:, 0] == cid_idx) & (full_labels[:, 1] == op_id)
    return full_coords[mask]


# =============================================================
# 主程式：从 capsid_full_with_labels.npz 中选点并画图
# =============================================================
if __name__ == "__main__":

    # 读取 capsid_full_with_labels
    data = np.load("data/biomole/capsid_full_with_labels.npz")
    full_coords = data["coords"]
    full_labels = data["labels"]

    print("[INFO] Loaded capsid_full_with_labels.npz")

    # 想要的 target copies
    targets = [
        ("B", 1), ("B", 6),
        ("C", 1), ("C", 6),
        ("D", 1), ("D", 6),
    ]

    # 颜色组：同链用相近色
    color_map = {
        "B1": "#08306b",   # 深蓝
        "B6": "#4292c6",   # 浅蓝
        "C1": "#67000d",   # 深红
        "C6": "#fb6a6a",   # 浅红
        "D1": "#00441b",   # 深绿
        "D6": "#41ab5d",   # 浅绿
    }

    # -------------------------------------------------------------
    # 整理选出的 copy
    # -------------------------------------------------------------
    selected_data = []
    for chain, op_id in targets:
        coords = select_copy(full_coords, None, full_labels, chain, op_id)
        key = f"{chain}{op_id}"
        selected_data.append((key, coords))
        print(f"[INFO] {key}: {coords.shape[0]} atoms")

    # -------------------------------------------------------------
    # 绘制 3D 图
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for key, coords in selected_data:
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            s=2,
            alpha=0.55,
            color=color_map[key],
            label=key,
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend(title="Copies")
    ax.set_title("Selected Copies from B, C, D (color-coded by chain)")

    plt.tight_layout()
    plt.show()
