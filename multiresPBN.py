import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
from scipy.ndimage import zoom
import seaborn as sns

# # ==========================================================
# # 1️⃣ 点云 → 冲激图（双线性分布）
# # ==========================================================
# def splat_points_to_grid(points, xlim, ylim, nx, ny, dtype=np.float32):
#     """
#     把点云双线性映射到规则网格上形成冲激图 I[y,x].
#     points: (N, 2) array
#     """
#     x0, x1 = xlim
#     y0, y1 = ylim
#     dx = (x1 - x0) / (nx - 1)
#     dy = (y1 - y0) / (ny - 1)

#     I = np.zeros((ny, nx), dtype=dtype)

#     # 连续坐标 → 网格索引
#     fx = (points[:, 0] - x0) / dx
#     fy = (points[:, 1] - y0) / dy
#     xL = np.floor(fx).astype(int)
#     yL = np.floor(fy).astype(int)
#     xH = xL + 1
#     yH = yL + 1

#     # 双线性权重
#     wxH = fx - xL
#     wyH = fy - yL
#     wxL = 1.0 - wxH
#     wyL = 1.0 - wyH

#     def add_safe(y, x, w):
#         mask = (x >= 0) & (x < nx) & (y >= 0) & (y < ny)
#         if np.any(mask):
#             np.add.at(I, (y[mask], x[mask]), w[mask].astype(dtype))

#     add_safe(yL, xL, wxL * wyL)
#     add_safe(yL, xH, wxH * wyL)
#     add_safe(yH, xL, wxL * wyH)
#     add_safe(yH, xH, wxH * wyH)
#     return I


# # ==========================================================
# # 2️⃣ 频域高斯核
# # ==========================================================
# def gaussian_filter_hat(nx, ny, dx, dy, sigma):
#     """
#     频域高斯核 Ĝ(kx, ky) = exp(-0.5 * sigma^2 * (kx^2 + ky^2))
#     """
#     kx = 2.0 * np.pi * np.fft.rfftfreq(nx, d=dx)
#     ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
#     KX, KY = np.meshgrid(kx, ky, indexing="xy")
#     return np.exp(-0.5 * sigma**2 * (KX**2 + KY**2), dtype=np.float32)


# # ==========================================================
# # 3️⃣ κ=2 情况的多分辨率场生成 (FFT)
# # ==========================================================
# def generate_fields_fft_kappa2(points, xlim, ylim, etas, nx=1201, ny=1201, dtype=np.float32):
#     x0, x1 = xlim
#     y0, y1 = ylim
#     dx = (x1 - x0) / (nx - 1)
#     dy = (y1 - y0) / (ny - 1)

#     # 1) 点云 → 冲激图
#     I = splat_points_to_grid(points, xlim, ylim, nx, ny, dtype=dtype)

#     # 2) 一次 FFT
#     I_hat = np.fft.rfft2(I)

#     # 3) 不同 η 的场
#     F_stack = np.empty((len(etas), ny, nx), dtype=dtype)
#     for i, eta in enumerate(etas):
#         sigma = float(eta) / np.sqrt(2.0)
#         G_hat = gaussian_filter_hat(nx, ny, dx, dy, sigma)
#         F_eta = np.fft.irfft2(I_hat * G_hat, s=(ny, nx))
#         F_stack[i] = F_eta
#     return F_stack


# ===========

def precompute_distance_squared(points, xlim, ylim, nx, ny):
    """
    为所有点与所有网格预先计算距离平方 D_i(x,y)
    返回 D: shape (n_points, ny, nx)
    """
    x0, x1 = xlim
    y0, y1 = ylim

    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    X, Y = np.meshgrid(xs, ys)

    # flatten grid for vectorization
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)   # (ny*nx, 2)

    D_list = []
    for px, py in points:
        dx = grid[:,0] - px
        dy = grid[:,1] - py
        D_list.append(dx*dx + dy*dy)

    D = np.stack(D_list, axis=0)    # (n_points, ny*nx)
    D = D.reshape(len(points), ny, nx)

    return D  # shape (N_points, ny, nx)


def compute_field_for_eta(D, eta):
    """
    使用预计算的距离平方 D 来计算某个 η 的场
    D shape = (n_points, ny, nx)
    """
    if eta == 0:
        raise ValueError("eta 不能为 0")

    return np.exp(- D / (eta*eta)).sum(axis=0)


def compute_fields(points, xlim, ylim, nx, ny, etas):
    """
    主函数：先预计算距离；再对每个 η 生成场
    返回：F_stack shape = (n_eta, ny, nx)
    """
    D = precompute_distance_squared(points, xlim, ylim, nx, ny)

    F_stack = np.zeros((len(etas), ny, nx), dtype=np.float32)

    for i, eta in enumerate(etas):
        F_stack[i] = compute_field_for_eta(D, eta)

    return F_stack


# ==========================================================
# 4️⃣ 归一化 + superlevel 选择
# ==========================================================
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
    max_per_field = F_stack.max(axis=(1, 2), keepdims=True)  # shape (n_eta, 1, 1)

    # 按公式归一化：1 - F/F.max()
    F_norm = 1.0 - F_stack / max_per_field
    F_norm = np.clip(F_norm, 1e-8, 1 - 1e-8)
    return F_norm

# ==========================================================
# 5️⃣ 持久区间 + PBN 计算
# ==========================================================
def build_cubical_intervals(field, k=0, homology_p=2):
    Cubical = getattr(gd, "CubicalComplex", None)
    if Cubical is None:
        from gudhi.cubical_complex import CubicalComplex as Cubical
    cc = Cubical(top_dimensional_cells=field)
    cc.persistence(homology_coeff_field=homology_p, min_persistence=0.0)
    return np.array(cc.persistence_intervals_in_dimension(k), dtype=float)


def compute_betti_curve_from_intervals(intervals, t_grid):
    if intervals.size == 0:
        return np.zeros_like(t_grid, dtype=int)
    b = intervals[:, 0][:, None]
    d = intervals[:, 1][:, None]
    t = t_grid[None, :]
    alive = (b <= t) & (t < d)
    return alive.sum(axis=0).astype(int)


def assemble_pbn_matrix(F_stack, etas, k=0, n_t=256):
    vals = F_stack.ravel()
    t_min, t_max = vals.min(), vals.max()
    t_grid = np.linspace(t_min, t_max, n_t)
    B = np.zeros((len(etas), n_t), dtype=int)
    for i, F in enumerate(F_stack):
        intervals = build_cubical_intervals(F, k=k)
        B[i, :] = compute_betti_curve_from_intervals(intervals, t_grid)
    return etas, t_grid, B


# ==========================================================
# 6️⃣ 可视化
# ==========================================================
# def plot_pbn_heatmap(
#     etas, t_grid, B, 
#     eta_label=r"Resolution $\eta$", 
#     t_label=r"Threshold $t$",
#     title=None
# ):
#     """
#     绘制 log10(B+1) 的 PBN 热力图，不包含等高线。
#     """

#     # ---- 1) 对所有 PBN 加 1 然后取 log10 ----
#     B_log = np.log10(B + 1.0)   # 以 10 为底的对数

#     fig, ax = plt.subplots(figsize=(6, 4))

#     # imshow extent
#     extent = (t_grid[0], t_grid[-1], etas[0], etas[-1])

#     # ---- 2) 用 log 值作为颜色画图 ----
#     im = ax.imshow(
#         B_log,
#         extent=extent,
#         origin="lower",
#         aspect="auto",
#         interpolation="bicubic",
#         cmap=cmap,
#     )

#     # ---- 3) colorbar 显示 log10 值 ----
#     cbar = fig.colorbar(im, ax=ax)
#     cbar.set_label(r"$\log_{10}(\beta_k(t) + 1)$")

#     # ---- 4) 坐标轴与标题 ----
#     ax.set_xlabel(t_label)
#     ax.set_ylabel(eta_label)
#     if title:
#         ax.set_title(title)

#     fig.tight_layout()
#     plt.show()
#     return fig, ax


def plot_pbn_heatmap(
    etas, t_grid, B,
    eta_label=r"Resolution $\eta$",
    t_label=r"Threshold $t$",
    title=None,
    interp_factor=8,        # 插值倍数（越大越平滑）
    levels=2000              # contourf 等高线层数（越多越平滑）
):
    """
    绘制平滑版 PBN 图：
    1. 对 B 进行 log10(B+1)
    2. 进行二维 cubic 插值（zoom）
    3. 用 contourf 绘制（连续平滑，不带像素感）

    输入：
        etas: shape (n_eta,)
        t_grid: shape (n_t,)
        B: shape (n_eta, n_t)
    """

    # ----- 1) log10(B+1) -----
    B_log = np.log10(B + 1.0)

    # ----- 2) 插值 2D 数组，使热图极度平滑 -----
    # zoom=(eta方向, t方向)
    B_smooth = zoom(B_log, zoom=(interp_factor, interp_factor), order=3)

    # 新的高分辨率网格
    eta_hi = np.linspace(etas.min(), etas.max(), B_smooth.shape[0])
    t_hi   = np.linspace(t_grid.min(), t_grid.max(), B_smooth.shape[1])

    # ----- 3) 使用 icefire colormap -----
    #cmap = sns.color_palette("icefire", as_cmap=True)

    # ----- 4) 绘图：contourf，让图像像连续函数一样 -----
    fig, ax = plt.subplots(figsize=(6, 4))

    ctf = ax.contourf(
        t_hi, eta_hi, B_smooth,
        levels=levels,
        cmap="RdYlBu"
    )

    # ----- 5) colorbar -----
    cbar = fig.colorbar(ctf, ax=ax)
    cbar.set_label(r"$\log_{10}(\beta_k(t)+1)$")

    # 每 0.5 一个刻度
    zmin = B_log.min()
    zmax = B_log.max()
    ticks = np.arange(np.floor(zmin), np.ceil(zmax), 0.5)
    cbar.set_ticks(ticks)

    # ----- 6) 轴标与标题 -----
    ax.set_xlabel(t_label)
    ax.set_ylabel(eta_label)
    if title:
        ax.set_title(title)

    fig.tight_layout()
    plt.show()

    return fig, ax


# ==========================================================
# 7️⃣ 主函数示例
# ==========================================================
if __name__ == "__main__":
    # 加载点云
    points = np.loadtxt("data/hex_points.txt", comments="#")
    xlim, ylim = (-30, 30), (-30, 30)

    # 分辨率网格（连续变化）
    etas = np.linspace(0.05, 10.0, 200)

    # --- FFT 场生成 ---
    F_stack = compute_fields(points, xlim, ylim, nx=601, ny=601, etas=etas,)

    # --- 归一化 & superlevel ---
    F_norm = normalize_fields(F_stack)

    # --- PBN (k=0, 1) ---
    for k in [0, 1]:
        etas, t_grid, B = assemble_pbn_matrix(F_norm, etas, k=k)
        # --- 画图 ---
        plot_pbn_heatmap(etas, t_grid, B, title=fr"PBN Heatmap ($k={k}$)")
