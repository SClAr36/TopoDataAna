import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

# 1️ 读取 C60 点云
points = np.loadtxt("c60.xyz", skiprows=1, usecols=(2, 3, 4))
print(f"共 {len(points)} 个原子")

# 2️ 构建 Rips 复形
max_edge_length = 6.0   # 最大边长（Å）
rips = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
st = rips.create_simplex_tree(max_dimension=3)
print(f"Simplices: {st.num_simplices()}, Dimension: {st.dimension()}")

# 3️ 获取 filtration 信息
filtration = list(st.get_filtration())
values = [a for _, a in filtration]
print(f"Filtration 值范围: [{min(values):.2f}, {max(values):.2f}]")

# 4️ 定义绘图函数
def plot_filtration(points, st, eps, ax):
    ax.scatter(points[:,0], points[:,1], points[:,2], s=20, c='black')
    for simplex, val in st.get_filtration():
        if len(simplex) == 2 and val < eps:
            i, j = simplex
            ax.plot(*zip(points[i], points[j]), c='gray', lw=0.6)
    ax.set_title(f"ε < {eps:.2f}")
    ax.set_box_aspect([1,1,1])
    ax.axis('off')

# 5️ 绘制不同阶段的过滤结构
fig = plt.figure(figsize=(12,4))
eps_values = [1, 2, 3]
for k, eps in enumerate(eps_values, 1):
    ax = fig.add_subplot(1, 3, k, projection='3d')
    plot_filtration(points, st, eps, ax)
plt.suptitle("Rips Filtration of C60 Molecule", fontsize=14)
plt.show()
