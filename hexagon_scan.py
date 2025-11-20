import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def select_copy(full_coords, full_labels, chain, op_id):
    chain_map = {"A": 0, "B": 1, "C": 2, "D": 3}
    cid_idx = chain_map[chain]
    mask = (full_labels[:, 0] == cid_idx) & (full_labels[:, 1] == op_id)
    return full_coords[mask]


def plot_six_copies(selected_dict, color_map):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for key, coords in selected_dict.items():
        ax.scatter(
            coords[:, 0], coords[:, 1], coords[:, 2],
            s=2, alpha=0.55, color=color_map[key], label=key
        )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Scanning for Hexagon – current Bk, Ck, Dk")
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # ---------------------------------------------------------
    # 读取 capsid_full_with_labels
    # ---------------------------------------------------------
    data = np.load("data/biomole/capsid_full_with_labels.npz")
    full_coords = data["coords"]
    full_labels = data["labels"]
    print("[INFO] Loaded capsid_full_with_labels.npz")

    # ---------------------------------------------------------
    # 固定三点：B1, C1, D1
    # ---------------------------------------------------------
    fixed_keys = ["B1", "C1", "D1"]
    fixed_coords = {
        "B1": select_copy(full_coords, full_labels, "B", 1),
        "C1": select_copy(full_coords, full_labels, "C", 1),
        "D1": select_copy(full_coords, full_labels, "D", 1),
    }

    # 颜色方案（链系一致）
    color_map = {
        "B1": "#08306b", "B2": "#4292c6",
        "C1": "#67000d", "C2": "#fb6a6a",
        "D1": "#00441b", "D2": "#41ab5d",
    }

    # ---------------------------------------------------------
    # 扫描 Bi, Ci, Di for i = 2..60
    # ---------------------------------------------------------
    for k in range(2, 61):
        print(f"\n[SCAN] Trying B{k}, C{k}, D{k} ...")

        # 动态更新颜色 (B/C/D 不同链, 但亮度根据 k 自动变化)
        color_map[f"B{k}"] = "#4292c6"
        color_map[f"C{k}"] = "#fb6a6a"
        color_map[f"D{k}"] = "#41ab5d"

        selected = {
            "B1": fixed_coords["B1"],
            "C1": fixed_coords["C1"],
            "D1": fixed_coords["D1"],
            f"B{k}": select_copy(full_coords, full_labels, "B", k),
            f"C{k}": select_copy(full_coords, full_labels, "C", k),
            f"D{k}": select_copy(full_coords, full_labels, "D", k),
        }

        # 显示 3D 图
        plot_six_copies(selected, color_map)

        input(f"[PAUSE] 当前为 B{k}, C{k}, D{k}。按 Enter 继续下一个 ...")
