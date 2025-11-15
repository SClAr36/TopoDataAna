import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load point cloud of C60 fullerene
points = np.loadtxt("C60-Ih.xyz", skiprows=2, usecols=(1, 2, 3))
print(f"Loaded {len(points)} atoms")

# 2. Compute distance matrix
dist_mat = distance_matrix(points, points)
print("Distance matrix shape:", dist_mat.shape)

# 3. Compute correlation matrix using Gaussian kernel
eta = 6.0  # 调节核宽度 (Å)
kappa = 2.0  # 调节核形状
corr_mat = np.exp(-(dist_mat / eta)**kappa)
mod_corr_mat = 1 - corr_mat

# 4. Plot the correlation matrix
fig, ax = plt.subplots(figsize=(6, 4))
im = ax.imshow(mod_corr_mat, cmap="OrRd", interpolation="nearest")
plt.colorbar(im, ax=ax, label="Correlation strength")
ax.set_title("C60 Exponential Kernel Correlation Matrix")
ax.set_xlabel("Atom index")
ax.set_ylabel("Atom index")
plt.tight_layout()
plt.savefig("cor_mat0.png", dpi=300)
plt.show()


# 3) 按阈值二值化，画图
thresholds = [0.1, 0.3, 0.5]
fig, axes = plt.subplots(1, len(thresholds), figsize=(12, 4), squeeze=False)
axes = axes.ravel()

for ax, d in zip(axes, thresholds):
    # adjacency[i,j] = 1 代表 dist[i,j] < d 的连通（包含对角线）
    adjacency = (mod_corr_mat <= d).astype(int)

    im = ax.imshow(adjacency, vmin=0, vmax=1, cmap="Blues", interpolation="nearest")
    ax.set_title(f"Filtration by distance d = {d} Å")
    ax.set_xlabel("Atom index")
    ax.set_ylabel("Atom index")

#fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="1 = connected (dist <= d)")
plt.tight_layout()
plt.savefig("filt0.png", dpi=300)
plt.show()