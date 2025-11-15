import numpy as np
import matplotlib.pyplot as plt

def hexagon(center: tuple[float, float], edge: float) -> np.ndarray:
    """Return 6 vertices of a hexagon centered at (x0, y0) with edge length."""
    x0, y0 = center
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]  # 6 vertices
    x = x0 + edge * np.cos(angles)
    y = y0 + edge * np.sin(angles)
    return np.stack([x, y], axis=1)

def build_multires_hex(levels: int = 3, L1: float = 20.0) -> np.ndarray:
    """Each level replaces every center with a hexagon of smaller size."""
    centers = np.array([[0.0, 0.0]])  # initial center

    for i in range(levels):
        edge = L1 * (0.25 ** i)
        new_centers = np.empty((0, 2))
        for c in centers:
            hex_pts = hexagon(tuple(c), edge)
            # 直接替换，不使用 append
            new_centers = np.vstack([new_centers, hex_pts])
        centers = new_centers  # replace centers for next level

    return centers

points = build_multires_hex(levels=3)
np.savetxt("hex_points.txt", points, fmt="%.6f", header="x y")

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(points[:, 0], points[:, 1], s=20, c='darkorange', edgecolors='k')
ax.set_aspect('equal', 'box')
ax.set_title('Multiresolution Hexagonal Fractal (3 levels)')
ax.axis('on')
plt.savefig("hex_fractal.png", dpi=300)
plt.show()

print(f"Total nodes: {len(points)}")

