import numpy as np
import matplotlib.pyplot as plt


def compute_field(points: np.ndarray, xlim, ylim, nx=200, ny=200, eta=1.0, kappa=2.0):
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

def plot_multiresolution(points, xlim, ylim, etas, nx=200, ny=200, save_data: bool = False):
    """
    Evaluate and plot f(r) for multiple eta values.
    """
    fig, axes = plt.subplots(1, len(etas), figsize=(4 * len(etas), 4), squeeze=False)
    axes = axes.ravel()
    
    for ax, eta in zip(axes, etas):
        X, Y, F = compute_field(points, xlim, ylim, nx=nx, ny=ny, eta=eta)
        im = ax.imshow(F, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                       origin='lower', cmap='viridis')
        #ax.scatter(points[:,0], points[:,1], s=10, c='r', alpha=0.6)
        ax.set_title(f"$\\eta={eta}$")
        plt.colorbar(im, ax=ax, shrink=0.7)

        # ====== 高效保存每个分辨率结果 ======
        if save_data:
            np.savez_compressed(
                f"data/multires/field_eta{eta:.1f}.npz",
                X=X, Y=Y, F=F,
                eta=eta, xlim=xlim, ylim=ylim,
                nx=nx, ny=ny
            )
    
    plt.tight_layout()
    plt.savefig("multires.png", dpi=500)
    plt.show()

# ===== Example usage =====
# Suppose you already have a 216-point cloud
# Example: load your data (replace this with your real file)
points = np.loadtxt("data/hex_points.txt", comments="#")
# Define the total area
xlim = (-30, 30)
ylim = (-30, 30)

# Define your multiresolution list
#etas = [0.5, 1.0, 2.0, 4.0]
etasmin = [0.2, 0.4, 0.6, 1.0]
etasmax = [3.0, 4.0, 10.0, 30.0]
plot_multiresolution(points, xlim, ylim, etasmin, nx=1201, ny=1201, save_data=True)
