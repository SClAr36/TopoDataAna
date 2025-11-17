import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import seaborn as sns

import gudhi as gd


def precompute_D(points, xlim, ylim, nx, ny):
    """
    预计算距离平方:
    返回 D: shape = (n_points, ny, nx)
    """
    xs = np.linspace(xlim[0], xlim[1], nx, dtype=np.float32)
    ys = np.linspace(ylim[0], ylim[1], ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)

    D = []
    for px, py in points:
        D.append((X - px)**2 + (Y - py)**2)

    return np.stack(D, axis=0).astype(np.float32)

@njit(parallel=True, fastmath=True)
def compute_field_eta_numba(D, eta):
    """
    对某个 eta 计算高斯场：
    D shape = (n_points, ny, nx)
    返回 F shape = (ny, nx)
    """
    n_points, ny, nx = D.shape
    F = np.zeros((ny, nx), dtype=np.float32)
    inv_eta2 = 1.0 / (eta * eta)

    for j in prange(ny):
        for i in range(nx):
            s = 0.0
            for p in range(n_points):
                s += np.exp(- D[p, j, i] * inv_eta2)
            F[j, i] = s

    return F

def compute_fields(points, xlim, ylim, nx, ny, etas):
    """
    主函数：先预计算 D，再对每个 η 求场。
    返回 F_stack: shape = (n_eta, ny, nx)
    """
    print("Precomputing distance matrix D ...")
    D = precompute_D(points, xlim, ylim, nx, ny)

    F_stack = np.zeros((len(etas), ny, nx), dtype=np.float32)

    print("Computing fields for each eta ...")
    for i, eta in enumerate(etas):
        F_stack[i] = compute_field_eta_numba(D, float(eta))
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(etas)} done.")
    
    print("Finished computing all fields.\n")
    np.save
    return F_stack

def normalize_fields(F_stack):
    """
    对每个场单独归一化：
        F_norm = 1 - F / F.max()
    不进行裁剪，不做 superlevel 取负。
    
    输入:
        F_stack: shape (n_etas, ny, nx)
    
    输出:
        F_norm: 同形状，每个场归一化到 [0,1]
    """
    # 对每个 η 的场分别求最大值，保留维度方便广播
    print("Normalizing fields ...")
    max_per_field = F_stack.max(axis=(1, 2), keepdims=True)  # shape (n_eta, 1, 1)

    # 按公式归一化：1 - F/F.max()
    F_norm = 1.0 - F_stack / max_per_field
    F_norm = np.clip(F_norm, 1e-6, 1 - 1e-6)
    print("Normalization done.\n")
    return F_norm

def build_intervals(field, k=0, homology_p=2):
    """
    使用 Gudhi 的 CubicalComplex 计算给定场的持久性区间（维度 k）。

    参数：
        field: 2D array, shape (ny, nx)
        k: 计算哪一维的持久同调（0,1,...）
        homology_p: 同调系数域（缺省为 Z2）

    返回：
        intervals: numpy array, shape (n_intervals, 2)
    """
    cc = gd.CubicalComplex(top_dimensional_cells=field)
    cc.persistence(homology_coeff_field=homology_p, min_persistence=0.0)
    return np.array(cc.persistence_intervals_in_dimension(k), dtype=float)

def betti_curve(intervals, t_grid):
    if len(intervals) == 0:
        return np.zeros_like(t_grid, dtype=int)

    b = intervals[:,0][:,None]
    d = intervals[:,1][:,None]
    t = t_grid[None,:]

    return ((b <= t) & (t < d)).sum(axis=0)

def compute_pbn_matrix(F_norm, etas, k=0, n_t=1001):
    """
    返回：
        etas               shape (n_eta)
        t_grid             shape (n_t)
        B (Betti matrix)   shape (n_eta, n_t)
    """
    print("Computing PBN matrix ...")
    vals = F_norm.ravel()
    t_min, t_max = vals.min(), vals.max()

    t_grid = np.linspace(t_min, t_max, n_t)
    B = np.zeros((len(etas), n_t), dtype=np.int32)

    for i in range(len(etas)):
        intervals = build_intervals(F_norm[i], k=k)
        B[i] = betti_curve(intervals, t_grid)
        if (i+1) % 10 == 0:
            print(f"  PBN: {i+1}/{len(etas)} done")

    print("PBN matrix computed.\n")
    return etas, t_grid, B

def save_pbn_data(filename, etas, t_grid, B):
    np.savez_compressed(filename, etas=etas, t_grid=t_grid, B=B)
    print(f"Saved PBN data to {filename}")

def load_pbn_data(filename):
    data = np.load(filename)
    return data["etas"], data["t_grid"], data["B"]

def plot_pbn_heatmap(
    etas, t_grid, B, 
    eta_label=r"Resolution $\eta$", 
    t_label=r"Threshold $t$",
    title=None,
    save_dir=None,
):
    """
    绘制 log10(B+1) 的 PBN 热力图，不包含等高线。
    """

    # ---- 1) 对所有 PBN 加 1 然后取 log10 ----
    B_log = np.log10(B + 1.0)   # 以 10 为底的对数

    fig, ax = plt.subplots(figsize=(6, 4))

    # imshow extent
    extent = (t_grid[0], t_grid[-1], etas[0], etas[-1])

    # ---- 2) 用 log 值作为颜色画图 ----
    im = ax.imshow(
        B_log,
        extent=extent,
        origin="lower",
        aspect="auto",
        interpolation="spline16",
        cmap="RdYlBu_r",
    )

    # ---- 3) colorbar 显示 log10 值 ----
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}(\beta_k(t) + 1)$")

    # ---- 4) 坐标轴与标题 ----
    ax.set_xlabel(t_label)
    ax.set_ylabel(eta_label)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    if save_dir:
        plt.savefig(save_dir, dpi=500)
    plt.show()
    return fig, ax


if __name__ == "__main__":
    # # Example usage
    points = np.loadtxt("data/hex_points.txt", comments="#")
    xlim, ylim = (-30, 30), (-30, 30)
    nx = ny = 1201
    etas = np.linspace(0.01, 10, 1000)

    # # 1) 生成场
    # F_stack = compute_fields(points, xlim, ylim, nx, ny, etas)
    
    # # 2) 归一化
    # F_norm = normalize_fields(F_stack)
    
    # np.savez_compressed(
    #     "data/multires/fields_eta0-10(1000etas).npz",
    #     F_stack=F_stack,
    #     F_norm=F_norm,
    #     etas=etas
    # )
    # print("Saved to fields_eta.npz")

    data = np.load("data/multires/fields_eta0-10(1000etas).npz")
    F_norm  = data["F_norm"]

    for k in [0, 1]:
        # 3) PBN 计算
        etas, t_grid, B = compute_pbn_matrix(F_norm, etas, k=k)
        
        # 4) 保存
        #save_pbn_data(f"pbn_eta{len(etas)}res{nx}t{len(t_grid)}({k=}).npz", etas, t_grid, B)
        
        # 5) 可视化
        plot_pbn_heatmap(etas, t_grid, B)#, save_dir=f"figs/hexagonal_frac/pbn_k{k}eta{len(etas)}t{t_grid}.png",)
    # for k in [0, 1]:
    #     etas, t_grid, B = load_pbn_data(f"pbn_eta1000res1201(k={k}).npz")
    #     plot_pbn_heatmap(etas, t_grid, B)#, save_dir=f"figs/hexagonal_frac/pbn_heatmap_k{k}(eta{len(etas)}).png",)