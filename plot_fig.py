import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ====== 1. 读取点云 ======
points = np.loadtxt("c60.xyz", skiprows=1, usecols=(2, 3, 4))
print(f"共 {len(points)} 个原子")

# ====== 2. 生成球坐标模板 ======
def sphere_mesh(center, r, resolution=20):
    """生成一个球面网格点 (x, y, z)，以 center 为圆心，r 为半径"""
    u = np.linspace(0, 2*np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + r * np.outer(np.cos(u), np.sin(v))
    y = center[1] + r * np.outer(np.sin(u), np.sin(v))
    z = center[2] + r * np.outer(np.ones_like(u), np.cos(v))
    return x, y, z

# ====== 3. 绘制函数 ======
def plot_rips_growth(points, radius, ax):
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(f"Radius = {radius:.2f}")
    ax.axis('off')

    # 绘制球体
    for p in points:
        X, Y, Z = sphere_mesh(p, radius)
        ax.plot_surface(X, Y, Z, color='skyblue', alpha=0.15, linewidth=0.01)

    # 绘制点
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='black', s=10)

    # 绘制相交的边
    n = len(points)
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(points[i] - points[j])
            if dist < 2 * radius:   # 球体相交
                ax.plot(*zip(points[i], points[j]), c='gray', lw=0.7, alpha=1.0)

# ====== 4. 多阶段绘制 ======
fig = plt.figure(figsize=(12, 4))
radii = [0.0, 1.5, 3.0]
for k, r in enumerate(radii, 1):
    ax = fig.add_subplot(1, 3, k, projection='3d')
    plot_rips_growth(points, r, ax)

plt.suptitle("Growth of Rips Balls around C60 Points", fontsize=14)
plt.show()


from matplotlib.animation import FuncAnimation

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

def update(frame):
    ax.cla()
    plot_rips_growth(points, radius=frame, ax=ax)
    return ax,

ani = FuncAnimation(fig, update, frames=np.linspace(0.0, 3.0, 16), interval=10)
plt.show()
