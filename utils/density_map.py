import numpy as np
import matplotlib.pyplot as plt


def rigidity_density_map_noweight(points: np.ndarray, xlim, ylim, nx=200, ny=200, eta=1.0, kappa=2.0):
    """
    points: (N, 2) array
    xlim, ylim: (xmin, xmax), (ymin, ymax)
    nx, ny: grid resolution
    eta: resolution parameter
    kappa: shape parameter (default 2)
    """
    # Build meshgrid
    xs = np.linspace(xlim[0], xlim[1], nx)
    ys = np.linspace(ylim[0], ylim[1], ny)
    X, Y = np.meshgrid(xs, ys)
    grid = np.stack([X.ravel(), Y.ravel()], axis=1)  # (M, 2)
    
    # Pairwise distances: (M, N)
    diff = grid[:, None, :] - points[None, :, :]
    r = np.linalg.norm(diff, axis=2)
    
    # Compute Φ and f
    phi = np.exp(-(r / eta)**kappa)
    denom = phi.sum(axis=1)
    mu = np.maximum(denom, 1e-12)
    
    return X, Y, mu.reshape(ny, nx)

def rigidity_density_map_3d(
    points: np.ndarray,
    xlim,
    ylim,
    zlim,
    grid_spacing: float = 0.3,
    eta: float = 1.0,
    kappa: float = 2.0,
    weights: np.ndarray | None = None,
):
    """
    计算 3D FRI 密度场：

        rho(x) = sum_j w_j * exp( - (||x - r_j|| / eta)^kappa )

    参数
    ----
    points : (N, 3) 原子坐标（Å）
    xlim, ylim, zlim : (min, max) 三个方向的边界（Å）
    grid_spacing : 网格间距（Å），建议 0.3
    eta : FRI 分辨率参数（Å）
    kappa : 形状参数，论文常用 2
    weights : (N,) 权重 w_j，例如原子序数；如果 None 则默认为 1

    返回
    ----
    xs, ys, zs : 三个 1D 坐标轴
    rho : shape (nx, ny, nz) 的密度场
    """
    points = np.asarray(points, dtype=np.float32)
    N = points.shape[0]

    if weights is None:
        weights = np.ones(N, dtype=np.float32)
    else:
        weights = np.asarray(weights, dtype=np.float32)
        assert weights.shape == (N,)

    # 构造格点坐标轴
    xs = np.arange(xlim[0], xlim[1] + grid_spacing * 0.5, grid_spacing, dtype=np.float32)
    ys = np.arange(ylim[0], ylim[1] + grid_spacing * 0.5, grid_spacing, dtype=np.float32)
    zs = np.arange(zlim[0], zlim[1] + grid_spacing * 0.5, grid_spacing, dtype=np.float32)

    nx, ny, nz = len(xs), len(ys), len(zs)
    print(f"Grid shape: ({nx}, {ny}, {nz}), total voxels = {nx*ny*nz:.3e}")

    rho = np.zeros((nx, ny, nz), dtype=np.float32)

    # 预先算好 eta^2 或 eta^kappa，避免在循环里反复算
    if abs(kappa - 2.0) < 1e-8:
        inv_eta2 = 1.0 / (eta * eta)

    for j in range(N):
        px, py, pz = points[j]
        w = weights[j]

        dx2 = (xs - px) ** 2           # (nx,)
        dy2 = (ys - py) ** 2           # (ny,)
        dz2 = (zs - pz) ** 2           # (nz,)

        # 广播得到 r^2(x,y,z) = dx2 + dy2 + dz2
        r2 = dx2[:, None, None] + dy2[None, :, None] + dz2[None, None, :]

        if abs(kappa - 2.0) < 1e-8:
            phi = np.exp(-r2 * inv_eta2)  # exp(- r^2 / eta^2)
        else:
            # 一般 kappa 的情况：r^kappa = (r^2)^{kappa/2}
            r_k = r2 ** (0.5 * kappa)
            phi = np.exp(-r_k / (eta ** kappa))

        rho += w * phi

    return xs, ys, zs, rho
